import torch
import torch.nn.functional as F
import torch.nn as nn



class LatentDiffusion(nn.Module):
    def __init__(
            self,
            diffuser_model,
            first_stage_model,
            cond_stage_model,
            image_size,
            image_channel,
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
        self.first_stage_model_proj = nn.Conv2d(first_stage_model.latent_chann, cond_stage_c, 1)

        #instantiate cond stage
        self.cond_stage_model = cond_stage_model
        self.cond_stage_model.eval()
        for param in self.cond_stage_model.parameters():
            param.requires_grad = False

        # self.cond_ids = torch.full(size=(self.timesteps,),
        #                            fill_value=self.timesteps-1,
        #                            dtype=torch.long)
        # ids = torch.round(torch.linspace(0, self.timesteps-1, num_timesteps_cond)).long()
        # self.cond_ids[:self.num_timesteps_cond] = ids

    def configure_optimziers(self, lr):
        params = list(self.diffuser_model.parameters())
        if self.cond_stage_trainable:
            params = params + list(self.cond_stage_model.parameters())

        optimizer = torch.optim.AdamW(params, lr=lr)
        return optimizer

    @torch.no_grad()
    def run_first_stage(self, img, cond, cond_mask):
        z = self.first_stage_model(img, mode='encode')[0] #[b,c,h,w]

        c = self.cond_stage_model(
            input_ids=cond,
            attention_mask=cond_mask).last_hidden_state  ##[b,l,c]

        return [z, c]
    
    @torch.no_grad()
    def decode_first_stage(self, samples):
        img = self.first_stage_model(samples, mode='decode')

        return img

    def forward(self, img, cond, cond_mask, mode='full'):
        if mode=='decode':
            # img: [4,], int
            # cond: [b,l], int
            # cond_mask: [b,l], bool
            cond = self.cond_stage_model(
                input_ids=cond,
                attention_mask=cond_mask).last_hidden_state  ##[b,l,c])

            z = self.diffuser_model.sample(img, cond)
            samples = self.decode_first_stage(z)[0]
            return samples
        
        
        # img: [b,c,h,w], float
        # cond: [b,l], int
        b, c, h, w = img.shape
        self.device = img.device

        z_latent, cond_latent = self.run_first_stage(img, cond, cond_mask)

        t = torch.randint(0, self.diffuser_model.timesteps, (b, ), device=self.device).long()

        return self.diffuser_model.p_losses(z_latent, t, cond_latent)
