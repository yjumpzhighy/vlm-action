import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.cuda.amp import autocast

from random import random
from tqdm.auto import tqdm


def linear_beta_scheduler(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.view(b, *((1, )) * (len(x_shape) - 1))  #expand to (b,1,1,1)


class GaussianDiffusion(nn.Module):
    def __init__(self,
                 model,
                 image_size,
                 image_channel,
                 timesteps,
                 ddim_sampling_steps=None,
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
        self.self_condition = model.self_condition if hasattr(
            model, "self_condition") else None
        self.objective = objective
        self.device = device

        # increasing noise var in each step
        betas = linear_beta_scheduler(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        # first 1.0 to start, and remove last
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        log_one_minus_alphas_cumprod = torch.log(1. - alphas_cumprod)
        sqrt_reciprocal_alphas_cumprod = torch.sqrt(1. / alphas_cumprod)
        sqrt_reciprocal_minus_one_alphas_cumprod = torch.sqrt(1. /
                                                              alphas_cumprod -
                                                              1)

        posterior_var = betas * (1. - alphas_cumprod_prev) / (1. -
                                                              alphas_cumprod)
        posterior_log_var_clipped = torch.log(posterior_var.clamp(min=1e-20))
        posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (
            1. - alphas_cumprod)
        posterior_mean_coef2 = (1. - alphas_cumprod_prev
                                ) * torch.sqrt(alphas) / (1. - alphas_cumprod)

        self.register_buffer('betas', betas.to(torch.float32))
        self.register_buffer('alphas', alphas.to(torch.float32))
        self.register_buffer('alphas_cumprod',
                             alphas_cumprod.to(torch.float32))
        self.register_buffer('alphas_cumprod_prev',
                             alphas_cumprod_prev.to(torch.float32))
        self.register_buffer('sqrt_alphas_cumprod',
                             sqrt_alphas_cumprod.to(torch.float32))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             sqrt_one_minus_alphas_cumprod.to(torch.float32))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             log_one_minus_alphas_cumprod.to(torch.float32))
        self.register_buffer('sqrt_reciprocal_alphas_cumprod',
                             sqrt_reciprocal_alphas_cumprod.to(torch.float32))
        self.register_buffer(
            'sqrt_reciprocal_minus_one_alphas_cumprod',
            sqrt_reciprocal_minus_one_alphas_cumprod.to(torch.float32))
        self.register_buffer('posterior_var', posterior_var.to(torch.float32))
        self.register_buffer('posterior_log_var_clipped',
                             posterior_log_var_clipped.to(torch.float32))
        self.register_buffer('posterior_mean_coef1',
                             posterior_mean_coef1.to(torch.float32))
        self.register_buffer('posterior_mean_coef2',
                             posterior_mean_coef2.to(torch.float32))

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
        return img * 0.5 + 0.5

    @autocast(enabled=False)
    def q_sample(self, x0, t, noise):
        # xt = sqrt(a1a2..at)*x0 + sqrt(1-a1a2..at)*e
        out = extract(self.sqrt_alphas_cumprod, t, x0.shape) * x0 + \
               extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise
        return out

    def model_prediction(self, xt, t, x_self_cond=None, clip_x0=False, rederive_pred_noise=False):
        model_out = self.model(xt, t, x_self_cond)
        #maybe_clip = torch.clamp(-1,1) if clip_x0 else t

        if self.objective == 'pred_noise':
            pred_noise = model_out
            # predict x0 from noise at t
            pred_x0 = extract(self.sqrt_reciprocal_alphas_cumprod,t,xt.shape) * xt - \
                  extract(self.sqrt_reciprocal_minus_one_alphas_cumprod,t,xt.shape)*pred_noise

            if clip_x0:
                torch.clamp_(pred_x0, -1, 1)
                
            # recalculate noise rather than direct model output
            # if clip_x0 and rederive_pred_noise:
            #     pred_noise = (extract(self.sqrt_reciprocal_alphas_cumprod,t,xt.shape) * xt - pred_x0) / \
            #        extract(self.sqrt_reciprocal_minus_one_alphas_cumprod,t,xt.shape)
                
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

    def p_losses(self, x0, t):
        b, c, h, w = x0.shape
        noise = torch.randn_like(x0, device=x0.device)  # N(0,I) distribution

        # forward from x0 to xt
        xt = self.q_sample(x0, t, noise)  #[b,c,h,w]

        # predicted x0 from xt
        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_prediction(xt, t)[1]
                x_self_cond.detach_()

        model_out = self.model(xt, t, x_self_cond)

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

        loss = F.mse_loss(model_out, target, reduction='none')
        #loss = torch.mean(loss, dim=(1,2,3))
        #loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, img):
        # img normalized from 0~1
        b, c, h, w = img.shape
        self.device = img.device

        t = torch.randint(0, self.timesteps, (b, ), device=self.device).long()
        img = self.normalize(img)
        loss = self.p_losses(img, t)
        return loss

    def sample(self, batch_size=1):
        img = torch.randn(
            (batch_size, self.image_channel, self.image_size, self.image_size),
            device=self.device)

        x0 = None
        for t in tqdm(reversed(range(0, self.timesteps))):
            x_self_cond = x0 if self.self_condition else None
            img, x0 = self.p_sample(img, t, x_self_cond)

        img = self.unnormalize_img(img)
        return img

    def p_sample(self, xt, t, x_self_cond):
        b, *_ = xt.shape
        device = xt.device
        # [t,t,t,...,t]
        batched_times = torch.full((b, ), t, device=device, dtype=torch.long)

        model_mean, model_logvar, x0 = self.p_mean_var(xt,
                                                       batched_times,
                                                       x_self_cond,
                                                       clip_denoise=True)
        noise = torch.randn_like(xt, device=device) if t > 0 else 0.
        pred_img = model_mean + (0.5 * model_logvar).exp() * noise
        return pred_img, x0

    def p_mean_var(self, xt, t, x_self_cond, clip_denoise=True):
        preds = self.model_prediction(xt, t, x_self_cond)
        x0 = preds[1]

        if clip_denoise:
            x0.clamp_(-1., 1.)

        model_mean, posterior_logvar = self.q_posterior(x0, xt, t)
        return model_mean, posterior_logvar, x0

    def q_posterior(self, x0, xt, t):
        posterior_mean = extract(self.posterior_mean_coef1, t, xt.shape) * x0 + \
                          extract(self.posterior_mean_coef2, t, xt.shape) * xt

        posterior_logvar = extract(self.posterior_log_var_clipped, t, xt.shape)
        return posterior_mean, posterior_logvar

    def sample_ddim(self, batch_size=1):
        times = torch.linspace(-1,self.timesteps-1,steps=self.ddim_sampling_steps+1) 
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1],times[1:])) #[[cur,next]]
        
        img = torch.randn(
            (batch_size, self.image_channel, self.image_size, self.image_size),
            device=self.device)
        
        pred_x0 = None
        for time,time_next in tqdm(time_pairs):
            times_cond = torch.full((batch_size, ), time, 
                                        device=self.device, dtype=torch.long)
            self_cond = pred_x0 if self.self_condition else None
            pred_noise, pred_x0 = self.model_prediction(img, times_cond, self_cond, True)

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
                

