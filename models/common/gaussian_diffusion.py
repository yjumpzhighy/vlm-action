import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.cuda.amp import autocast

from random import random
from tqdm.auto import tqdm

import numpy as np
from functools import partial


def linear_beta_scheduler(timesteps):
    linear_start = 0.0015
    linear_end = 0.0155
    # return torch.linspace(beta_start, beta_end, timesteps)
    betas = (
                torch.linspace(linear_start ** 0.5, linear_end ** 0.5, timesteps, dtype=torch.float64) ** 2
        )
    return betas.numpy()


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1, )) * (len(x_shape) - 1))  #expand to (b,1,1,1)


class GaussianDiffusion(nn.Module):
    def __init__(self,
                 model,
                 image_size,
                 image_channel,
                 timesteps,
                 ddim_sampling_steps=50,
                 ddim_sampling_eta=1.,
                 objective="pred_noise",
                 device=None):
        super().__init__()
        self.model = model
        self.image_size = image_size
        self.image_channel = image_channel
        self.timesteps = timesteps
        self.ddim_sampling_steps = ddim_sampling_steps
        self.ddim_sampling_eta = ddim_sampling_eta
        self.objective = objective
        self.device = device

        # increasing noise var in each step
        betas = linear_beta_scheduler(timesteps)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        # first 1.0 to start, and remove last
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_reciprocal_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_reciprocal_minus_one_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod) 
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_var', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_var_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))


        self.normalize = self.normalize_img

        signal_noise_ratio = self.alphas_cumprod / (1 - self.alphas_cumprod)
        loss_weight = signal_noise_ratio / (signal_noise_ratio + 1)
        self.register_buffer('loss_weight', loss_weight.to(torch.float32))

    def normalize_img(self, img):
        # img is normalized form 0-255 to 0-1
        # normalize to [-1,1]
        return 2 * img - 1

    def unnormalize_img(self, img):
        # unnormalize to [0,1]
        #return (img + 1) * 0.5
        return torch.clamp((img+1.0)/2.0, min=0.0, max=1.0)


    def q_sample(self, x0, t, noise):
        # xt = sqrt(a1a2..at)*x0 + sqrt(1-a1a2..at)*e
        out = extract(self.sqrt_alphas_cumprod, t, x0.shape) * x0 + \
               extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise
        return out

    def unet_predict(self, xt, t, cond=None):
        return self.model(xt,t,cond)


    def model_prediction(self, xt, t, cond=None, clip_denoised=False):
        model_out = self.unet_predict(xt, t, cond)
        #maybe_clip = torch.clamp(-1,1) if clip_x0 else t

        if self.objective == 'pred_noise':
            pred_noise = model_out
            # predict x0 from noise at t
            pred_x0 = extract(self.sqrt_reciprocal_alphas_cumprod,t,xt.shape) * xt - \
                  extract(self.sqrt_reciprocal_minus_one_alphas_cumprod,t,xt.shape)*pred_noise

            if clip_denoised:
                pred_x0.clamp_(-1., 1.)
                
        elif self.objective == 'pred_x0':
            pred_x0 = model_out
            if clip_x0:
                torch.clamp_(pred_x0, -1, 1)
            # predict noise at t from x0
            pred_noise = (extract(self.sqrt_reciprocal_alphas_cumprod,t,xt.shape) * xt - pred_x0) / \
                   extract(self.sqrt_reciprocal_minus_one_alphas_cumprod,t,xt.shape)
        elif self.objective == "pred_v":
            v = model_out
            pred_x0 = extract(self.sqrt_alphas_cumprod, t, xt.shape) * xt - \
                        extract(self.sqrt_one_minus_alphas_cumprod, t, xt.shape) * v
            if clip_x0:
                torch.clamp_(pred_x0, -1, 1)
            pred_noise = (extract(self.sqrt_reciprocal_alphas_cumprod,t,xt.shape) * xt - pred_x0) / \
                   extract(self.sqrt_reciprocal_minus_one_alphas_cumprod,t,xt.shape)
        else:
            raise ValueError("type not supported")
        return pred_noise, pred_x0

    def p_losses(self, x0, t, cond=None):
        b, c, h, w = x0.shape
        noise = torch.randn_like(x0, device=x0.device)  # N(0,I) distribution

        # forward from x0 to xt
        xt = self.q_sample(x0, t, noise)  #[b,c,h,w]

        model_out = self.unet_predict(xt, t, cond)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == "pred_x0":
            target = x0
        elif self.objective == "pred_v":
            # "Progressive Distill for fast sampling of diffusion models",
            # v is formulated as combination of x0 and noise, kinding of
            # use the model to predict both x0 and noise
            # v = sqrt(a1a2..at)*noise - sqrt(1-a1a2..at)*x0
            target = extract(self.sqrt_alphas_cumprod,t,x0.shape) * noise - \
                    extract(self.sqrt_one_minus_alphas_cumprod,t,x0.shape) * x0
        else:
            raise ValueError("type not supported")

        loss = F.mse_loss(model_out, target)
        #loss = torch.mean(loss, dim=(1,2,3))
        #loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss

    def forward(self, img, cond=None):
        # img normalized from 0~1
        b, c, h, w = img.shape
        self.device = img.device

        t = torch.randint(0, self.timesteps, (b, ), device=self.device).long()
        #img = self.normalize(img)
        loss = self.p_losses(img, t, cond)
        return loss

    def sample(self, img, cond=None):
        for t in tqdm(reversed(range(0, self.timesteps))):
            img = self.p_sample(img, cond, t) 
        return img

    

    def p_mean_variance(self, x, c, t, clip_denoised: bool, return_x0=False):
        pred_x0 = self.model_prediction(x, t, c, clip_denoised=clip_denoised)[1]
        model_mean, posterior_logvar = self.q_posterior(pred_x0, x, t)
        return model_mean, posterior_logvar



    @torch.no_grad()
    def p_sample(self, x, c, t, clip_denoised=False, return_x0=False):
        b, *_ = x.shape
        device = x.device
        # [t,t,t,...,t]
        batched_times = torch.full((b, ), t, device=device, dtype=torch.long)

        model_mean, model_logvar= self.p_mean_variance(x=x, c=c, t=batched_times, 
                                                          clip_denoised=clip_denoised, 
                                                          return_x0=return_x0)
        noise = torch.randn_like(x, device=device) if t > 0 else 0.
        pred_img = model_mean + (0.5 * model_logvar).exp() * noise
        return pred_img


    def q_posterior(self, x0, xt, t):
        posterior_mean = extract(self.posterior_mean_coef1, t, xt.shape) * x0 + \
                          extract(self.posterior_mean_coef2, t, xt.shape) * xt

        posterior_logvar = extract(self.posterior_log_var_clipped, t, xt.shape)
        return posterior_mean, posterior_logvar

    def sample_ddim(self, img):
        times = torch.linspace(-1,self.timesteps-1,steps=self.ddim_sampling_steps+1) 
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1],times[1:])) #[[cur,next]]
        
        # img = torch.randn(
        #     (batch_size, self.image_channel, self.image_size, self.image_size),
        #     device=self.device)
        b, *_ = img.shape
        pred_x0 = None
        for time,time_next in tqdm(time_pairs):
            times_cond = torch.full((b, ), time, 
                                        device=self.device, dtype=torch.long)
            pred_noise, pred_x0 = self.model_prediction(img, times_cond, None, True)

            if time_next < 0:
                img = pred_x0
                continue
            
            accu_alpha = self.alphas_cumprod[time]
            accu_alpha_next = self.alphas_cumprod[time_next]
            
            sigma = self.ddim_sampling_eta * ((1-accu_alpha/accu_alpha_next) * \
                        (1-accu_alpha_next) / (1-accu_alpha)).sqrt()
            c = (1-accu_alpha_next-sigma**2).sqrt()
            
            noise = torch.randn_like(img)
            img = pred_x0 * accu_alpha_next.sqrt() + c * pred_noise + sigma * noise
            
        img = self.unnormalize_img(img)
        return img    
                

