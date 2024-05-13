import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from torch import optim
from scipy.stats import norm
import os
from torch.nn.parallel import DistributedDataParallel as DDP
#from torchpack import distributed as dist
import torch.distributed as dist
import math
from transformers import CLIPTokenizer, CLIPTextModel
from torch.cuda.amp import autocast, GradScaler

from common.datasets import ImageNetDataset, CifarDataset, MnistDataset
from common.variational_auto_encoder import VariationalAutoEncoder
from common.latent_diffusion import LatentDiffusion
from common.conditional_unet import ConditionalUNet
from common.gaussian_diffusion import GaussianDiffusion
from common.utils import init_setup, get_kl_loss, get_recon_loss, ddp_load_checkpoint, \
                         generate_latents_combinations, cosine_with_warmup_scheduler

def get_dataset(dataset_name, image_h, image_w):
    tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
    max_token_len = 64
    
    if dataset_name == 'imagenet':
        data_train = ImageNetDataset(
            os.path.abspath(os.path.join(os.getcwd(), '../../Data/imagenet')),
            IMAGE_SIZE,
            transforms.Compose([
                transforms.Resize((image_h, image_w)),
                transforms.ToTensor()
            ]), 'train', tokenizer, max_token_len)
    elif dataset_name == 'cifar':
        data_train = CifarDataset(
            os.path.abspath(os.path.join(os.getcwd(), '../../Data/cifar10')),
            IMAGE_SIZE,
            transforms.Compose([
                transforms.Resize((image_h, image_w)),
                transforms.ToTensor()
            ]), 'train', tokenizer, max_token_len)
    else:
        raise ValueError("dataset not supported.")
    return data_train


if __name__ == "__main__":
    init_setup()
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(dist.get_rank())

    FLAG_TRAIN_AUTOENCODER = False
    FLAG_TRAIN_LDM = True

    DATASET = 'cifar' #'cifar' #'imagenet
    BATCHSIZE = 512
    IMAGE_SIZE = 256
    IMAGE_C = 3
    EPOCHS = 10
    EMBEDDING_DIM = 64
    LATENT_DIM = 4
    SAVE_AUTOENCODER_PATH = 'data/stable_diff/autoencoder_imagenet256_ldm32_bat128_lat256_emb64_ep100.pth'

    data_train = get_dataset(DATASET, IMAGE_SIZE, IMAGE_SIZE)
    data_sampler = torch.utils.data.distributed.DistributedSampler(
            data_train,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=True)

    # step 1, train autoencoder
    autoencoder = VariationalAutoEncoder(IMAGE_SIZE,
                                         IMAGE_C,
                                         EMBEDDING_DIM,
                                         output_chann=IMAGE_C,
                                         latent_chann=LATENT_DIM,
                                         use_GAN=True).cuda()
    if FLAG_TRAIN_AUTOENCODER:
        model = torch.nn.parallel.DistributedDataParallel(
            autoencoder,
            device_ids=[dist.get_rank()],
            find_unused_parameters=True)

        train_loader = torch.utils.data.DataLoader(data_train,
                                                   batch_size=BATCHSIZE,
                                                   sampler=data_sampler,
                                                   num_workers=12,
                                                   pin_memory=True)

        encoder_optimizer = model.module.configure_optimziers(lr=5e-5)[0]
        discriminator_optimizer = model.module.configure_optimziers(
            lr=5e-5)[1]

        encoder_scheduler = cosine_with_warmup_scheduler(
            encoder_optimizer, EPOCHS, BATCHSIZE, len(train_loader.dataset),
            dist.get_world_size())
        discriminator_scheduler = cosine_with_warmup_scheduler(
            discriminator_optimizer, EPOCHS, BATCHSIZE,
            len(train_loader.dataset), dist.get_world_size())

        encoder_loss = 0.0
        discriminator_loss = 0.0
        encoder_scaler = GradScaler()
        discriminator_scaler = GradScaler()
        min_loss = np.finfo(np.float32).max
        optimizer_idx = 0

        total_gloabl_steps = len(
            train_loader) / dist.get_world_size() / BATCHSIZE * EPOCHS
        cur_global_steps = 0
        for epoch in range(EPOCHS):
            model.train()
            for idx, batch in enumerate(train_loader):
                cur_global_steps += 1

                x = batch['image']
                x = x.cuda()

                with autocast():
                    *_, loss, log = model(
                        x,
                        optimizer_idx=optimizer_idx,
                        gan_factor=1.0 if
                        cur_global_steps >= total_gloabl_steps * 0.2 else 0.0)

                if loss < 0:
                    print(log)

                if optimizer_idx == 0:
                    scheduler = encoder_scheduler
                    optimizer = encoder_optimizer
                    scaler = encoder_scaler
                    encoder_loss = loss.item()
                else:
                    scheduler = discriminator_scheduler
                    optimizer = discriminator_optimizer
                    scaler = discriminator_scaler
                    discriminator_loss = loss.item()

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                if discriminator_optimizer is not None:
                    optimizer_idx = (optimizer_idx + 1) % 2

            if dist.get_rank() == 0:
                print(
                    "Training autoencoder loss:",
                    encoder_loss,
                    "discriminator loss:",
                    discriminator_loss,
                    "epoch:",
                    epoch,
                    "lr:",
                    scheduler.get_last_lr())

            if dist.get_rank() == 0:
                torch.save(model.state_dict(), SAVE_AUTOENCODER_PATH)

    # if not os.path.exists(SAVE_AUTOENCODER_PATH):
    #     raise ValueError("autoencoder ckpt file not exists")
    # state_dict = ddp_load_checkpoint(SAVE_AUTOENCODER_PATH)
    # autoencoder.load_state_dict(state_dict, strict=True)

    if dist.get_rank() == 0:
        n = 3
        linspace = 10
        autoencoder.eval()
        figure = np.zeros((IMAGE_SIZE * n, IMAGE_SIZE * n, IMAGE_C))

        # grid mesh, values range from [-1,1]
        grids = [
            norm.ppf(np.linspace(0.05, 0.95, linspace))
            for i in range(autoencoder.encoded_image_h *
                           autoencoder.encoded_image_w * LATENT_DIM)
        ]

        with torch.no_grad():
            for i in range(n * n):
                z_sampled = np.array([
                    np.random.choice(grids[i])
                    for i in range(autoencoder.encoded_image_h *
                                   autoencoder.encoded_image_w * LATENT_DIM)
                ]).reshape(1, LATENT_DIM, autoencoder.encoded_image_h,
                           autoencoder.encoded_image_w)

                z_sampled = torch.FloatTensor(z_sampled).cuda()

                decode = autoencoder(z_sampled, mode='decode')[0]
                digit = decode[0].permute(
                    1, 2, 0).contiguous().detach().cpu().numpy()

                fig_row_idx = int(i / n)
                fig_col_idx = int(i % n)
                figure[fig_row_idx * IMAGE_SIZE:(fig_row_idx + 1) * IMAGE_SIZE,
                       fig_col_idx * IMAGE_SIZE:(fig_col_idx + 1) *
                       IMAGE_SIZE, :] = digit

        figure = (figure * 255).astype('uint8')
        if IMAGE_C == 1:
            figure = np.squeeze(figure, axis=-1)
            plt.imsave('encoder_image.png', figure, cmap='gray', format='png')
        elif IMAGE_C == 3:
            plt.imsave('encoder_image.png', figure, format='png')
        else:
            raise ValueError("image channels not supported.")



    # step 2, train ldm
    BATCHSIZE = 64
    EPOCHS = 10
    EMBEDDING_DIM = 64
    TIMESTEPS = 1000
    SAVE_LDM_PATH = 'data/stable_diff/ldm_cifar32_ldm16_emb64.pth'

    text_encoder = CLIPTextModel.from_pretrained(
        'openai/clip-vit-base-patch32')

    ldm = LatentDiffusion(GaussianDiffusion(ConditionalUNet(
        autoencoder.latent_image_h,
        autoencoder.latent_image_w,
        autoencoder.latent_chann,
        base_channel=EMBEDDING_DIM,
        output_channel=autoencoder.latent_chann,
        use_crossattn=True,
        context_c=text_encoder.config.hidden_size),
                                            autoencoder.latent_image_h,
                                            autoencoder.latent_chann,
                                            TIMESTEPS,
                                            objective='pred_noise',
                                            device=torch.device('cuda')),
                          autoencoder,
                          text_encoder,
                          autoencoder.encoded_image_h,
                          autoencoder.encoded_image_c,
                          num_timesteps_cond=1,
                          cond_stage_key='class_label',
                          cond_stage_trainable=False,
                          cond_stage_c=text_encoder.config.hidden_size,
                          conditioning_key='crossattn',
                          timesteps=1000,
                          device=torch.device('cuda')).cuda()

    if FLAG_TRAIN_LDM:
        model = torch.nn.parallel.DistributedDataParallel(
            ldm, device_ids=[dist.get_rank()], find_unused_parameters=True)

        train_loader = torch.utils.data.DataLoader(data_train,
                                                   batch_size=BATCHSIZE,
                                                   sampler=data_sampler,
                                                   num_workers=12,
                                                   pin_memory=True)

        optimizer = model.module.configure_optimziers(lr=1e-4)
        scheduler = cosine_with_warmup_scheduler(optimizer, EPOCHS, BATCHSIZE,
                                                 len(train_loader.dataset),
                                                 dist.get_world_size())

        cur_loss = 0.0
        min_loss = np.finfo(np.float32).max
        for epoch in range(EPOCHS):
            model.train()
            train_num = len(train_loader.dataset)
            for idx, batch in enumerate(train_loader):
                x = batch['image']
                c = batch['text_id']
                m = batch['attention_mask']
                # if 'label' in batch.keys():
                #     mask = (batch['label'] == 5)
                #     x = x[mask]
                #     if x.shape[0]==0:
                #         continue
                x = x.cuda()  #[b,c,h,w]
                c = c.cuda()  #[b,l]
                m = m.cuda()  #[b,l]

                loss = model(x, c, m)
                cur_loss = loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

            if dist.get_rank() == 0:
                print("Training ldm loss:", cur_loss, "epoch:", epoch, "lr:",
                      scheduler.get_last_lr())

            if dist.get_rank() == 0 and cur_loss < min_loss:
                min_loss = cur_loss
                torch.save(model.state_dict(), SAVE_LDM_PATH)

    
    if not os.path.exists(SAVE_LDM_PATH):
        raise ValueError("ldm ckpt file not exists")
    state_dict = ddp_load_checkpoint(SAVE_LDM_PATH)
    ldm.load_state_dict(state_dict, strict=True)
    ldm = torch.nn.parallel.DistributedDataParallel(
        ldm, device_ids=[dist.get_rank()], find_unused_parameters=False)

    if dist.get_rank() == 0:
        with torch.no_grad():
            sample_batch = 1
            cond, cond_mask = data_train.tokenize_text('shark, white shark')
            cond = torch.Tensor([cond]).int().cuda()
            cond_mask = torch.Tensor([cond_mask]).int().cuda()
            shape = [
                sample_batch, autoencoder.latent_chann,
                autoencoder.latent_image_h, autoencoder.latent_image_w
            ]

            samples = ldm(shape, cond, cond_mask, mode='decode')
            samples = samples.permute(0, 2, 3,
                                      1).contiguous().detach().cpu().numpy()

            figure = np.zeros((IMAGE_SIZE * 1, IMAGE_SIZE * 1, IMAGE_C))
            for i in range(sample_batch):
                fig_row_idx = int(i / sample_batch)
                fig_col_idx = int(i % sample_batch)
                figure[fig_row_idx * IMAGE_SIZE:(fig_row_idx + 1) * IMAGE_SIZE,
                       fig_col_idx * IMAGE_SIZE:(fig_col_idx + 1) *
                       IMAGE_SIZE, :] = samples[i]

            figure = (figure * 255).astype('uint8')
            if IMAGE_C == 1:
                figure = np.squeeze(figure, axis=-1)
                plt.imsave('ldm_image.png', figure, cmap='gray', format='png')
            elif IMAGE_C == 3:
                plt.imsave('ldm_image.png', figure, format='png')
            else:
                raise ValueError("image channels not supported.")
