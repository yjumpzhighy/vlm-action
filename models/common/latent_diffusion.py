import torch
import torch.nn.functional as F
import torch.nn as nn
import sys
import os
from tqdm.auto import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from omegaconf import OmegaConf
from latent_diffusion.ldm.util import instantiate_from_config
from latent_diffusion.ldm.models.diffusion.ddpm import LatentDiffusion as LatentDiffusion2

class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, key='class'):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)

    def forward(self, batch, key=None):
        if key is None:
            key = self.key
        # this is for use in crossattn
        c = batch[key][:, None]
        c = self.embedding(c)
        return c

class LatentDiffusion(nn.Module):
    def __init__(
            self,
            diffuser_model,
            first_stage_model,
            cond_stage_model,
            num_timesteps_cond=1,
            cond_stage_key='class_label',  #['caption','coord_bbox']
            cond_stage_trainable=False,
            cond_stage_c=512,
            concat_mode=True,
            cond_stage_forward=None,
            conditioning_key='crossattn',
            scale_factor=1.0,
            scale_by_std=False,
            timesteps=1000,
            device=None):
        super().__init__()
        
        self.num_timesteps_cond = num_timesteps_cond
        self.cond_stage_trainable = cond_stage_trainable
        self.conditioning_key = conditioning_key
        self.cond_stage_key = cond_stage_key
        self.scale_by_std = scale_by_std
        self.diffuser_model = diffuser_model

        #instantiate first stage
        self.first_stage_model = first_stage_model
        self.first_stage_model.eval()
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

        #instantiate cond stage
        self.cond_stage_model = cond_stage_model
        self.cond_stage_model.eval()
        for param in self.cond_stage_model.parameters():
            param.requires_grad = False

    def configure_optimziers(self, lr):
        params = list(self.diffuser_model.parameters()) #unet
        if self.cond_stage_trainable:
            params = params + list(self.cond_stage_model.parameters())

        optimizer = torch.optim.AdamW(params, lr=lr)
        return optimizer

    @torch.no_grad()
    def run_first_stage(self, img, cond, cond_mask):
        z = self.first_stage_model(img, mode='encode').sample() #[b,c,h,w]
        c = None
        if cond is not None:
            c = self.cond_stage_model(
                input_ids=cond,
                attention_mask=cond_mask).last_hidden_state  ##[b,l,c]

        return [z, c]
    
    @torch.no_grad()
    def decode_first_stage(self, samples):
        img = self.first_stage_model(samples, mode='decode')
        return img

    def forward(self, samples, cond=None, cond_mask=None, mode='full'):
        if mode=='decode':
            # img: list or tensor 
            # cond: [b,l], int
            # cond_mask: [b,l], bool
            if cond is not None:
                cond = self.cond_stage_model(
                    input_ids=cond,
                    attention_mask=cond_mask).last_hidden_state  ##[b,l,c])
            
            z = self.diffuser_model.sample(samples, cond)
            img = self.decode_first_stage(z)
            img = torch.clamp((img+1.0)/2.0, min=0.0, max=1.0)
            return img
            
        
        
        # img: [b,c,h,w], float
        # cond: [b,l], int
        b, c, h, w = samples.shape
        self.device = samples.device

        z_latent, cond_latent = self.run_first_stage(samples, cond, cond_mask)
        return self.diffuser_model(z_latent, cond_latent)

class MyLdm(nn.Module):
    def load_model_from_config(self, config, ckpt):
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt)#, map_location="cpu")
        sd = pl_sd["state_dict"]
        #model = instantiate_from_config(config.model)

        model = LatentDiffusion2(**(config.model.get("params", dict())))
        #model.load_state_dict(sd, strict=False)
        model.cuda()
        model.train()
        #model.eval()
        return model

    def get_model(self, config_file, ckpt_file):
        config = OmegaConf.load(config_file)  
        model = self.load_model_from_config(config, ckpt_file)
        return model

    def __init__(self):
        super().__init__()
        # self.model = AutoencoderKL.from_pretrained(
        #         "stabilityai/stable-diffusion-2-base",
        #         subfolder="vae",
        #         torch_dtype=torch.float32,
        #         resume_download=True)
        config_file = "/autox-dl/deeplearning/users/zuyuan/Projects/latent_diffusion/models/ldm/cin256/config.yaml"
        #config_file = "/autox-dl/deeplearning/users/zuyuan/Projects/latent_diffusion/configs/latent-diffusion/ldm-kl-8-backup.yaml"
        ckpt_file = "/autox-dl/deeplearning/users/zuyuan/Projects/latent_diffusion/models/ldm/cin256/model.ckpt"    
        self.model = self.get_model(config_file, ckpt_file)
        
        self.first_stage_model = None

        self.latent_image_h = self.model.image_size
        self.latent_image_w = self.model.image_size
        self.latent_chann = self.model.channels
        

    def forward(self, x, c=None, m=None, mode='full'):
        if mode == "decode":
            if c is not None:
                c = self.model.get_learned_conditioning({self.model.cond_stage_key: c.to(self.model.device)})

            for t in tqdm(reversed(range(0, 1000))):
                ts = torch.full((x.shape[0], ), t, device=x.device, dtype=torch.long)
                x = self.model.p_sample(x, c, t)

            x = self.model.decode_first_stage(x)
            x = torch.clamp((x+1.0)/2.0, min=0.0, max=1.0)
            return x


        if self.model.model.conditioning_key is not None:
            batch = {self.model.first_stage_key:x, self.model.cond_stage_key:c}
        else:
            batch = {self.model.first_stage_key:x}
            
        x, c = self.model.get_input(batch, self.model.first_stage_key)
        loss = self.model(x, c)[0]
        return loss


    def configure_optimziers(self, lr):
        # default train diffusion model and cond stage model
        return self.model.configure_optimizers(lr)
