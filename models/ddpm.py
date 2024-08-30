import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
import torch.distributed as dist
import os
from common.gaussian_diffusion import GaussianDiffusion
from common.conditional_unet import ConditionalUNet
from common.utils import ddp_load_checkpoint, cosine_with_warmup_scheduler, init_setup
from common.datasets import ImageNetDataset, ButterfliesDataset


if __name__ == "__main__":
    init_setup(seed=12345)
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(dist.get_rank())



    TIMESTEPS = 1000
    BATCHSIZE = 32
    IMAGE_SIZE = 128
    IMAGE_C = 3
    NUM_EPOCHS = 20
    EMBEDDING_BASE_DIM = 128
    EMBEDDING_MULTIS = (1,2,4,4)
    SAVE_MODEL_PATH = 'data/ddpm/myddpm_hfunet.pth'
    TRAIN_DDPM = True

    from transformers import CLIPTokenizer
    tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
    # data_train = ImageNetDataset(
    #         os.path.abspath(os.path.join(os.getcwd(), '../../Data/imagenet')),
    #         IMAGE_SIZE,
    #         transforms.Compose([
    #             transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    #             transforms.ToTensor(),
    #             transforms.Normalize([0.5], [0.5])
    #         ]), 'train[0:50000]', tokenizer, 64)
    
    data_train = ButterfliesDataset(
        os.path.abspath(os.path.join(os.getcwd(), '../../Data/butterflies')),
        IMAGE_SIZE,
        transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  #transform to [-1,1]
        ]),
        'train',
        tokenizer,
        64)

    data_sampler = torch.utils.data.distributed.DistributedSampler(
            data_train,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=True)

    # from diffusers import UNet2DModel
    # unet = UNet2DModel(
    #     sample_size=IMAGE_SIZE,  # the target image resolution
    #     in_channels=IMAGE_C,  # the number of input channels, 3 for RGB images
    #     out_channels=IMAGE_C,  # the number of output channels
    #     layers_per_block=2,  # how many ResNet layers to use per UNet block
    #     block_out_channels=(
    #         128, 256, 512, 512),  # the number of output channes for each UNet block
    #     down_block_types=(
    #         "DownBlock2D",  # a regular ResNet downsampling block
    #         "DownBlock2D",
    #         "DownBlock2D",
    #         "DownBlock2D",
    #         #"AttnDownBlock2D",
    #     ),
    #     up_block_types=(
    #         "UpBlock2D",  # a regular ResNet upsampling block
    #         #"AttnUpBlock2D",
    #         "UpBlock2D",
    #         "UpBlock2D",
    #         "UpBlock2D"),
    # ).cuda()

    unet = ConditionalUNet(IMAGE_SIZE,
                            IMAGE_SIZE,
                            IMAGE_C,
                            base_channel=EMBEDDING_BASE_DIM,
                            channel_mults=EMBEDDING_MULTIS,
                            output_channel=IMAGE_C,
                            add_down_up_skip=True).cuda()

    model = GaussianDiffusion(unet,
                                 IMAGE_SIZE,
                                 IMAGE_C,
                                 timesteps=TIMESTEPS,
                                 ddim_sampling_steps=20,
                                 objective='pred_noise',
                                 device=torch.device('cuda')).cuda()

    if TRAIN_DDPM:
        ddp_model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist.get_rank()],
            find_unused_parameters=True)

        train_loader = torch.utils.data.DataLoader(data_train,
                                                   batch_size=BATCHSIZE,
                                                   sampler=data_sampler,
                                                   num_workers=12,
                                                   pin_memory=True)

        optimizer = torch.optim.Adam(ddp_model.parameters(), lr=1e-4)

        from diffusers.optimization import get_cosine_schedule_with_warmup
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=50,
            num_training_steps=(len(train_loader) * NUM_EPOCHS),
        )

        total_step = len(train_loader)
        cur_loss = np.finfo(np.float32).max
        for epoch in range(NUM_EPOCHS):
            for idx, batch in enumerate(train_loader):
                x = batch['image'].cuda()

                loss = ddp_model(x)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()

            if dist.get_rank() == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                    epoch + 1, NUM_EPOCHS, idx + 1, total_step, loss.item()))

                # save best model
                if loss.item() < cur_loss:
                    cur_loss = loss.item()
                    torch.save(ddp_model.state_dict(), SAVE_MODEL_PATH)
                    print("===>Saved ckpt loss:", cur_loss)
    
    dist.barrier()
    state_dict = ddp_load_checkpoint(SAVE_MODEL_PATH, dist.get_rank())
    model.load_state_dict(state_dict, strict=True)

    if dist.get_rank() == 0:
        with torch.no_grad():
            img = torch.randn(
                (9, model.image_channel, model.image_size, model.image_size)).cuda()
            samples = model.sample(img)
            samples = samples.permute(
                0, 2, 3, 1).contiguous().detach().cpu().numpy()  #[b,h,w,c]

            figure = np.zeros((IMAGE_SIZE * 3, IMAGE_SIZE * 3, IMAGE_C))
            for i in range(9):
                fig_row_idx = int(i / 3)
                fig_col_idx = int(i % 3)
                figure[fig_row_idx * IMAGE_SIZE:(fig_row_idx + 1) * IMAGE_SIZE,
                    fig_col_idx * IMAGE_SIZE:(fig_col_idx + 1) *
                    IMAGE_SIZE, :] = samples[i]

            figure = (figure * 255).astype('uint8')
            if IMAGE_C == 1:
                figure = np.squeeze(figure, axis=-1)
                plt.imsave('ddpm_image.png', figure, cmap='gray', format='png')
            elif IMAGE_C == 3:
                plt.imsave('ddpm_image-multicrossattn_noskip.png', figure, format='png')
            else:
                raise ValueError("image channels not supported.")
