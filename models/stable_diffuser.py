import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from torch import optim
from scipy.stats import norm
import os
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import math
from transformers import CLIPTokenizer, CLIPTextModel
from torch.cuda.amp import autocast, GradScaler

from common.datasets import ImageNetDataset, CifarDataset, ButterfliesDataset
from common.variational_auto_encoder import VariationalAutoEncoder
from common.latent_diffusion import LatentDiffusion, MyLdm
from common.conditional_unet import ConditionalUNet
from common.gaussian_diffusion import GaussianDiffusion
from common.utils import init_setup, ddp_load_checkpoint, cosine_with_warmup_scheduler

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
            ]), 'train[0:1000]', tokenizer, max_token_len)
    elif dataset_name == 'cifar':
        data_train = CifarDataset(
            None,
            IMAGE_SIZE,
            transforms.Compose([
                transforms.Resize((image_h, image_w)),
                transforms.ToTensor()
            ]), 'train', tokenizer, max_token_len)
    elif dataset_name == 'butterfly':
        data_train = ButterfliesDataset(
        None,
        IMAGE_SIZE,
        transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  #transform to [-1,1]
        ]), 'train', tokenizer, 64)
    else:
        raise ValueError("dataset not supported.")
    return data_train


if __name__ == "__main__":
    init_setup(seed=379)
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(dist.get_rank())

    FLAG_TRAIN_AUTOENCODER = True
    FLAG_TRAIN_LDM = True

    DATASET = 'imagenet' #'cifar' 'imagenet 'butterfly'
    BATCHSIZE = 12
    IMAGE_SIZE = 256
    IMAGE_C = 3
    EPOCHS = 100
    EMBEDDING_BASE_DIM = 128
    EMBEDDING_MULTIS = (1,2,4,4) #donwsample fator 2^(len(multis)-1)
    LATENT_DIM = 3
    LEARNING_RATE = 5e-6
    SAVE_AUTOENCODER_PATH = 'data/stable_diff/autoencoderkl_imagenet256_emb128_lat4_0817.pth'
    SAVE_AUTOENCODER_LAST_PATH = SAVE_AUTOENCODER_PATH[:-4] + '_last.pth'


    data_train = get_dataset(DATASET, IMAGE_SIZE, IMAGE_SIZE)
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


    # step 1, train autoencoder
    unet = ConditionalUNet(IMAGE_SIZE,
                            IMAGE_SIZE,
                            IMAGE_C,
                            base_channel=EMBEDDING_BASE_DIM,
                            channel_mults=EMBEDDING_MULTIS,
                            output_channel=IMAGE_C,
                            add_down_up_skip=False).cuda()

    autoencoder = VariationalAutoEncoder(unet,
                                   IMAGE_SIZE,
                                   IMAGE_C,
                                   output_chann=IMAGE_C,
                                   latent_chann=LATENT_DIM,
                                   use_GAN=True,
                                   num_training_steps=(len(train_loader) * EPOCHS)).cuda()

    if FLAG_TRAIN_AUTOENCODER:
        ddp_model = torch.nn.parallel.DistributedDataParallel(
            autoencoder,
            device_ids=[dist.get_rank()],
            find_unused_parameters=True)

        encoder_optimizer = ddp_model.module.configure_optimziers(lr=LEARNING_RATE)[0]
        discriminator_optimizer = ddp_model.module.configure_optimziers(
            lr=LEARNING_RATE)[1]


        from diffusers.optimization import get_cosine_schedule_with_warmup
        encoder_lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=encoder_optimizer,
            num_warmup_steps=50,
            num_training_steps=(len(train_loader) * EPOCHS),
        )

        if discriminator_optimizer is not None:
            discriminator_lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=discriminator_optimizer,
                num_warmup_steps=50,
                num_training_steps=(len(train_loader) * EPOCHS),
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
        for epoch in range(EPOCHS):
            ddp_model.train()
            for idx, batch in enumerate(train_loader):
                x = batch['image']
                x = x.cuda()

                with autocast():
                    loss = ddp_model(
                        x,
                        optimizer_idx=optimizer_idx,
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
                
                if dist.get_rank() == 0 and global_step % 1000 == 0:
                    print(
                        epoch,
                        "/", 
                        EPOCHS,
                        ": autoencoder loss",
                        [val.item() for val in encoder_loss],
                        " /// discriminator loss",
                        [val.item() for val in discriminator_loss],
                    )
                    
            if dist.get_rank() == 0:
                print(epoch,
                    "/", 
                    EPOCHS,
                    ": autoencoder loss",
                    [val.item() for val in encoder_loss],
                    " /// discriminator loss",
                    [val.item() for val in discriminator_loss],
                )

                # best save
                if encoder_loss[0].item() < min_loss:
                    min_loss = encoder_loss[0].item()
                    torch.save(ddp_model.state_dict(), SAVE_AUTOENCODER_PATH)

                # last save
                if epoch == EPOCHS-1:
                    torch.save(ddp_model.state_dict(), SAVE_AUTOENCODER_LAST_PATH)
                    

    dist.barrier()
    if not os.path.exists(SAVE_AUTOENCODER_PATH):
        raise ValueError("autoencoder ckpt file not exists")
    state_dict = ddp_load_checkpoint(SAVE_AUTOENCODER_PATH, dist.get_rank())
    autoencoder.load_state_dict(state_dict, strict=True)

    if dist.get_rank() == 0:
        batch = data_train[333]
        figure = np.zeros((IMAGE_SIZE, IMAGE_SIZE, IMAGE_C))
        figure[:,:,:] = batch['image'].permute(1, 2, 0).contiguous().detach().cpu().numpy()
        figure = (figure * 255).astype('uint8')
        plt.imsave('vae_raw.png', figure, format='png')
        
        x = batch['image'].unsqueeze(0).cuda()
        z = autoencoder(x,mode='encode').sample()
        r = autoencoder(z,mode='decode')
        figure = np.zeros((IMAGE_SIZE, IMAGE_SIZE, IMAGE_C))
        figure[:,:,:] = r[0].permute(1, 2, 0).contiguous().detach().cpu().numpy()
        figure = (figure * 255).astype('uint8')
        plt.imsave('vae_reconstruct.png', figure, format='png')
    
    # step 2, train ldm
    BATCHSIZE = 12
    EPOCHS = 100
    LEARNING_RATE = 5e-5
    EMBEDDING_BASE_DIM = 128
    EMBEDDING_MULTIS = (1,2,4) #donwsample fator 2^(len(multis)-1)
    TIMESTEPS = 1000
    LDM_CONDITIONING = False
    SAVE_LDM_PATH = 'data/stable_diff/ldm_imagenet32_emb128_0817_labelcond.pth'
    SAVE_LDM_PATH_LAST = SAVE_LDM_PATH[:-4] + '_last.pth'

    text_encoder = CLIPTextModel.from_pretrained(
        'openai/clip-vit-base-patch32')

    unet = ConditionalUNet(
        autoencoder.latent_image_h,
        autoencoder.latent_image_w,
        autoencoder.latent_chann,
        base_channel=EMBEDDING_BASE_DIM,
        channel_mults=EMBEDDING_MULTIS,
        output_channel=autoencoder.latent_chann,
        add_down_up_skip=True,
        context_c = text_encoder.config.hidden_size if LDM_CONDITIONING else None
    )

    diffuser_model = GaussianDiffusion(unet,
                                        autoencoder.latent_image_h,
                                        autoencoder.latent_chann,
                                        TIMESTEPS,
                                        objective='pred_noise',
                                        device=torch.device('cuda'))

    ldm = LatentDiffusion(diffuser_model,
                          autoencoder,
                          text_encoder,
                          num_timesteps_cond=1,
                          cond_stage_key='class_label',
                          cond_stage_trainable=False,
                          cond_stage_c=text_encoder.config.hidden_size if LDM_CONDITIONING else None,
                          conditioning_key='crossattn',
                          timesteps=1000,
                          device=torch.device('cuda')).cuda()

    ldm = MyLdm().cuda()
    ldm.model.first_stage_model = autoencoder.cuda().eval()
    for param in ldm.model.first_stage_model.parameters():
        param.requires_grad = False

    if FLAG_TRAIN_LDM:
        model = torch.nn.parallel.DistributedDataParallel(
            ldm, device_ids=[dist.get_rank()], find_unused_parameters=True)

        tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')

        optimizer = model.module.configure_optimziers(LEARNING_RATE)
        scheduler = cosine_with_warmup_scheduler(optimizer, EPOCHS, BATCHSIZE,
                                                 len(train_loader.dataset),
                                                 dist.get_world_size())

        cur_loss = 0.0
        min_loss = np.finfo(np.float32).max
        scaler = GradScaler()
        global_step = 0
        for epoch in range(EPOCHS):
            # model.train()
            train_num = len(train_loader.dataset)
            for idx, batch in enumerate(train_loader):
                x = batch['image'].cuda() #[b,c,h,w]
                c = batch['text_id'].cuda() if LDM_CONDITIONING else None #[b,l]
                m = batch['attention_mask'].cuda() if LDM_CONDITIONING else None #[b,l] 
                
                
                c = batch['label'].cuda()
                
                #with autocast():
                loss = model(x, c, m)
                cur_loss = loss.item()

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                global_step += dist.get_world_size()
                if dist.get_rank() == 0 and global_step % 1000 == 0:
                    print("Training ldm loss:", cur_loss, "epoch:", epoch, "/", EPOCHS)
                    
            if dist.get_rank() == 0:
                print("Training ldm loss:", cur_loss, "epoch:", epoch, "/", EPOCHS, "lr:",
                        scheduler.get_last_lr())

                # best save    
                if cur_loss < min_loss:
                    min_loss = cur_loss
                    torch.save(model.state_dict(), SAVE_LDM_PATH)

                # last save
                if epoch == EPOCHS-1:
                    torch.save(model.state_dict(), SAVE_LDM_PATH_LAST)


    dist.barrier()
    if not os.path.exists(SAVE_LDM_PATH):
        raise ValueError("ldm ckpt file not exists")
    state_dict = ddp_load_checkpoint(SAVE_LDM_PATH, dist.get_rank())
    ldm.load_state_dict(state_dict, strict=True)

    if dist.get_rank() == 0:
        with torch.no_grad():
            n = 3
            sample_batch = n*n
            cond, cond_mask = data_train.tokenize_text('shark, white shark')
            cond = torch.Tensor([cond]).int().cuda()
            cond_mask = torch.Tensor([cond_mask]).int().cuda()
            
            cond, cond_mask = None, None
            
            
            latents = torch.randn(
                    (sample_batch, autoencoder.latent_chann,
                autoencoder.latent_image_h, autoencoder.latent_image_w)).cuda()
            
            cond = torch.tensor(sample_batch*[2]).cuda()
            samples = ldm(latents, cond, cond_mask, mode='decode')

            samples = samples.permute(0, 2, 3,
                                      1).contiguous().detach().cpu().numpy()

            figure = np.zeros((IMAGE_SIZE * n, IMAGE_SIZE * n, IMAGE_C))
            for i in range(sample_batch):
                fig_row_idx = int(i / n)
                fig_col_idx = int(i % n)
                figure[fig_row_idx * IMAGE_SIZE:(fig_row_idx + 1) * IMAGE_SIZE,
                    fig_col_idx * IMAGE_SIZE:(fig_col_idx + 1) *
                    IMAGE_SIZE, :] = samples[i]

            figure = (figure * 255).astype('uint8')
            if IMAGE_C == 1:
                figure = np.squeeze(figure, axis=-1)
                plt.imsave('ldm_image.png', figure, cmap='gray', format='png')
            elif IMAGE_C == 3:
                plt.imsave('ldm_image2.png', figure, format='png')
            else:
                raise ValueError("image channels not supported.")
