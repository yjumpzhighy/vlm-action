import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
import torch.distributed as dist
import os
from torch.cuda.amp import autocast, GradScaler

from common.variational_auto_encoder import VariationalAutoEncoder, MyAutoencoderKL
from common.conditional_unet import ConditionalUNet
from common.utils import ddp_load_checkpoint, cosine_with_warmup_scheduler, init_setup
from common.datasets import ImageNetDataset, ButterfliesDataset


if __name__ == "__main__":
    init_setup(seed=12345)
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(dist.get_rank())

    BATCHSIZE = 16
    IMAGE_SIZE = 256
    IMAGE_C = 3
    NUM_EPOCHS = 500
    EMBEDDING_BASE_DIM = 128
    EMBEDDING_MULTIS = (1,2,4,4)
    LATENT_DIM = 4
    LEARNING_RATE = 1e-5
    SAVE_MODEL_PATH = 'data/vae/vae_butterfly256_bs16_latent4_f8_embed128_ep500.pth'
    TRAIN_VAE = True

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
        'train', tokenizer, 64)

    data_sampler = torch.utils.data.distributed.DistributedSampler(
            data_train,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=True)

    train_loader = torch.utils.data.DataLoader(data_train,
                                                   batch_size=BATCHSIZE,
                                                   sampler=data_sampler,
                                                   num_workers=12,
                                                   pin_memory=True)

    unet = ConditionalUNet(IMAGE_SIZE,
                            IMAGE_SIZE,
                            IMAGE_C,
                            base_channel=EMBEDDING_BASE_DIM,
                            channel_mults=EMBEDDING_MULTIS,
                            output_channel=IMAGE_C,
                            add_down_up_skip=False).cuda()

    model = VariationalAutoEncoder(unet,
                                   IMAGE_SIZE,
                                   IMAGE_C,
                                   output_chann=IMAGE_C,
                                   latent_chann=LATENT_DIM,
                                   use_GAN=True).cuda()

    # model = MyAutoencoderKL().cuda()

                            

    if TRAIN_VAE:
        ddp_model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist.get_rank()],
            find_unused_parameters=True)

        encoder_optimizer = ddp_model.module.configure_optimziers(lr=LEARNING_RATE)[0]
        discriminator_optimizer = ddp_model.module.configure_optimziers(
            lr=LEARNING_RATE)[1]

        from diffusers.optimization import get_cosine_schedule_with_warmup
        encoder_lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=encoder_optimizer,
            num_warmup_steps=50,
            num_training_steps=(len(train_loader) * NUM_EPOCHS),
        )

        if discriminator_optimizer is not None:
            discriminator_lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=discriminator_optimizer,
                num_warmup_steps=50,
                num_training_steps=(len(train_loader) * NUM_EPOCHS),
            )
        else:
            discriminator_lr_scheduler = None

        encoder_loss = []
        discriminator_loss = []
        encoder_scaler = GradScaler()
        discriminator_scaler = GradScaler()
        min_loss = np.finfo(np.float32).max
        optimizer_idx = 0
        global_step = 0
        for epoch in range(NUM_EPOCHS):
            ddp_model.train()
            for idx, batch in enumerate(train_loader):
                x = batch['image']
                x = x.cuda()

                with autocast():
                    loss = ddp_model(
                        x,
                        optimizer_idx=optimizer_idx,
                        gan_factor=1.0 if (epoch >= int(NUM_EPOCHS*0.8)) else 0.0,
                        global_step=global_step)
                    main_loss = loss[0]

                if optimizer_idx == 0:
                    scheduler = encoder_lr_scheduler
                    optimizer = encoder_optimizer
                    scaler = encoder_scaler
                    encoder_loss = loss
                else:
                    scheduler = discriminator_lr_scheduler
                    optimizer = discriminator_optimizer
                    scaler = discriminator_scaler
                    discriminator_loss = loss

                optimizer.zero_grad()
                scaler.scale(main_loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                if discriminator_optimizer is not None:
                    optimizer_idx = (optimizer_idx + 1) % 2

                global_step += dist.get_world_size()

            if dist.get_rank() == 0:
                print(epoch,
                    "/", 
                    NUM_EPOCHS,
                    ": autoencoder loss",
                    [val.item() for val in encoder_loss],
                    " /// discriminator loss",
                    [val.item() for val in discriminator_loss],
                )

                if encoder_loss[0].item() < min_loss:
                    min_loss = encoder_loss[0].item()
                    torch.save(ddp_model.state_dict(), SAVE_MODEL_PATH)
    
    dist.barrier()
    state_dict = ddp_load_checkpoint(SAVE_MODEL_PATH, dist.get_rank())
    model.load_state_dict(state_dict, strict=True)


    if dist.get_rank() == 0:
        with torch.no_grad():

            data_iter = iter(train_loader)
            batch = next(data_iter)
            figure = np.zeros((IMAGE_SIZE, IMAGE_SIZE, IMAGE_C))
            figure[:,:,:] = batch['image'][0].permute(1, 2,
                                            0).contiguous().detach().cpu().numpy()
            figure = (figure * 255).astype('uint8')
            plt.imsave('vae_raw_image.png', figure, format='png')

            img = batch['image'].cuda()
            z_sampled = model(img, mode='encode').sample()

            img_reconstruct = model(z_sampled, mode='decode')
            figure = np.zeros((IMAGE_SIZE, IMAGE_SIZE, IMAGE_C))
            figure[:,:,:] = img_reconstruct[0].permute(1, 2,
                                            0).contiguous().detach().cpu().numpy()
            figure = (figure * 255).astype('uint8')
            plt.imsave('vae_reconstruct_image.png', figure, format='png')

    if dist.get_rank() == 0:
        with torch.no_grad():
            # z_sampled = torch.randn(
            #     (9, LATENT_DIM, model.encoded_image_h,
            #            model.encoded_image_w)).cuda()
            z_sampled = torch.randn(
                (9, 4, 32, 32)).cuda()
            samples = model(z_sampled, mode='decode')
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
                plt.imsave('vae_image.png', figure, cmap='gray', format='png')
            elif IMAGE_C == 3:
                plt.imsave('vae_image.png', figure, format='png')
            else:
                raise ValueError("image channels not supported.")



            
            

