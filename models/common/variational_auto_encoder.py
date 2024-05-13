import torch
from torch import nn
import torch.nn.functional as F
from torch import optim

import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.stats import norm
from functools import partial
from taming.modules.losses.vqperceptual import *

from .conditional_unet import ConditionalUNet


class VariationalAutoEncoder(nn.Module):
    def __init__(self,
                 image_size,
                 image_chann=3,
                 base_chann=64,
                 channel_mults=(1, 2, 4, 4),
                 output_chann=3,
                 latent_chann=3,
                 use_GAN=False):
        super().__init__()

        self.model = ConditionalUNet(image_size,
                                     image_size,
                                     image_chann,
                                     base_chann,
                                     output_channel=output_chann,
                                     short_cuts=False)

        self.encoded_image_h = self.model.encoded_image_h
        self.encoded_image_w = self.model.encoded_image_w
        self.encoded_image_c = self.model.encoded_image_c
        
        self.latent_image_h = self.model.encoded_image_h
        self.latent_image_w = self.model.encoded_image_w
        self.latent_chann = latent_chann
        
        self.mean_logvar = nn.Conv2d(self.encoded_image_c, 2*latent_chann, 1)
        self.decode_projection = nn.Conv2d(latent_chann, self.encoded_image_c, 1)
        
        # self.mean_linear = nn.Linear(
        #     self.encoded_image_c * self.encoded_image_h * self.encoded_image_w,
        #     latent_chann)
        # self.logvar_linear = nn.Linear(
        #     self.encoded_image_c * self.encoded_image_h * self.encoded_image_w,
        #     latent_chann)
        # self.decode_projection = nn.Linear(
        #     latent_chann,
        #     self.encoded_image_c * self.encoded_image_h * self.encoded_image_w)
        self.out_sigmoid = nn.Sigmoid()

        self.use_GAN = use_GAN
        if use_GAN:
            self.discriminator = NLayerDiscriminator(
                    input_nc=output_chann, n_layers=3,
                    use_actnorm=False).apply(weights_init)
        else:
            self.discriminator = None
        self.discriminator_loss = hinge_d_loss
        self.discriminator_bce_loss = nn.BCEWithLogitsLoss()

        self.perceptual_loss = LPIPS().eval()
        self.recon_loss_regulizer = nn.Parameter(torch.zeros(size=()))

    def reparameter(self, mean, logvar):
        eps = torch.randn_like(mean)  #random sample from (0,1)
        return mean + torch.exp(
            logvar * 0.5) * eps  # u + sqrt(var)*eps ~ N(u, var)

    def forward(self, x, t=None, cond=None, mode='full', optimizer_idx=0, gan_factor=0.0): 
        if mode == 'decode':
            return self.decode_latent(x), None, None, None
            
            # if x.dim()==2: #[b,z]
            #     return self.decode_latent(x), None, None, None
            # elif x.dim()==4: #[b,c,h,w]
            #     return self.decode_latent_img(x), None, None, None
            # else:
            #     raise ValueError("vae decode latent dim not supported.")
                

        if mode == 'encode':
            return self.encode_image(x), None, None, None

        b, c, h, w = x.shape
        d = x.device

        # encoded = self.model.encode(x, t, cond).view(b, -1)
        # mean = self.mean_linear(encoded)  #[b,latent_chann]
        # logvar = self.logvar_linear(encoded)  #[b,latent_chann]
        # z = self.reparameter(mean, logvar)  #[b,latent_chann]
        # recon_x = self.decode_projection(z).view(b, self.encoded_image_c,
        #                                          self.encoded_image_h,
        #                                          self.encoded_image_w)

        encoded = self.model.encode(x, t, cond) #[b,encoded_chann,h',w']
        mean_logvar = self.mean_logvar(encoded) #[b,2*encoded_chann,h',w']
        mean, logvar = torch.chunk(mean_logvar, 2, dim=1) 
        z = self.reparameter(mean, logvar) #[b,latent_chann,h',w']

        recon_x = self.decode_projection(z) #[b,encoded_chann,h,w]
        recon_x = self.model.decode(recon_x) #[b,c,h,w]
        recon_x = self.out_sigmoid(recon_x)  #[b,c,h,w]

        loss, log = self.get_loss(mean, logvar, x, recon_x, optimizer_idx=optimizer_idx, 
                                  gan_factor=gan_factor)

        return recon_x, mean, logvar, loss, log

    def decode_latent(self, z):
        #b, _ = z.shape
        recon_x = self.decode_projection(z)
        recon_x = self.model.decode(recon_x)
        recon_x = self.out_sigmoid(recon_x)  #[b,c,h,w]
        return recon_x
    
    def decode_latent_img(self, z_img):
        recon_x = self.model.decode(z_img)
        recon_x = self.out_sigmoid(recon_x)  #[b,c,h,w]
        return recon_x

    def encode_image(self, x, t=None, cond=None):
        # b, c, h, w = x.shape
        # encoded = self.model.encode(x, t, cond).view(b, -1)
        # mean = self.mean_linear(encoded)  #[b,latent_chann]
        # logvar = self.logvar_linear(encoded)  #[b,latent_chann]
        # z = self.reparameter(mean, logvar)  #[b,latent_chann]
        # z = self.decode_projection(z).view(b, self.encoded_image_c,
        #                                    self.encoded_image_h,
        #                                    self.encoded_image_w)

        encoded = self.model.encode(x, t, cond) #[b,encoded_chann,h',w']
        mean_logvar = self.mean_logvar(encoded) #[b,2*encoded_chann,h',w']
        mean, logvar = torch.chunk(mean_logvar, 2, dim=1) 
        z = self.reparameter(mean, logvar) #[b,latent_chann,h',w']

        return z

    def get_kl_loss(self, mean, logvar):
        kl_loss = torch.mean(
            -0.5 * torch.sum(1 + logvar - mean.pow(2) - torch.exp(logvar), dim=[1,2,3]),
            0)

        return kl_loss

    def get_recon_loss(
        self,
        x,
        recon_x,
    ):
        # recon_loss = torch.mean(torch.sum(F.binary_cross_entropy(recon_x,x,reduction='none'),
        #                                   dim=(1,2,3)))
        # recon_loss = torch.mean(torch.sum(F.mse_loss(recon_x,x,reduction='none'),
        #                                   dim=(1,2,3)))

        recon_loss = torch.abs(x.contiguous() - recon_x.contiguous())
        recon_loss = torch.sum(recon_loss) / recon_loss.shape[0]
        return recon_loss

    def configure_optimziers(self, lr):
        optim_autoencoder = torch.optim.Adam(
            list(self.model.parameters()) +
            list(self.mean_logvar.parameters()) +
            # list(self.mean_linear.parameters()) +
            # list(self.logvar_linear.parameters()) +
            list(self.decode_projection.parameters()),
            lr=lr,
            betas=(0.5, 0.9))
        if self.use_GAN:
            optim_discriminator = torch.optim.Adam(
                self.discriminator.parameters(), lr=lr, betas=(0.5, 0.9))
        else:
            optim_discriminator = None

        return [optim_autoencoder, optim_discriminator]

    def get_loss(self, mean, logvar, x, recon_x, optimizer_idx, gan_factor=0.0, cond=None):

        l1_recon_loss = torch.abs(x.contiguous() - recon_x.contiguous())
        perceptual_recon_loss = self.perceptual_loss(x.contiguous(),
                                                     recon_x.contiguous())

        recon_loss = l1_recon_loss * 1.0 + perceptual_recon_loss * 1.0
        nll_loss = recon_loss / torch.exp(
            self.recon_loss_regulizer) + self.recon_loss_regulizer
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]

        # nll_loss = torch.sum(
        #     l1_recon_loss) / l1_recon_loss.shape[0] + torch.sum(
        #         perceptual_recon_loss) / perceptual_recon_loss.shape[0]

        kl_loss = self.get_kl_loss(mean, logvar) * 0.01

        if not self.use_GAN:
            loss = nll_loss + kl_loss
            log = {"loss:": loss.clone().detach(),
                   "nll loss:": nll_loss.clone().detach(),
                   "kl_loss:": kl_loss.clone().detach()
            }
            return loss, log
        
        if optimizer_idx == 0:
            # update autoencoder generator
            if cond is None:
                logits_fake = self.discriminator(recon_x.contiguous())
            else:
                logits_fake = self.discriminator(
                    torch.cat((recon_x.contiguous(), cond), dim=1))

            # bce(logits_fake, 0)
            # gan_loss = F.binary_cross_entropy_with_logits(logits_fake, torch.zeros_like(logits_fake,
            #                                                                             device=logits_fake.device))
            
            gan_loss = -torch.mean(logits_fake)

            gan_weight = self.calculate_adaptive_weight(nll_loss, gan_loss)
            # gan_factor = adopt_weight(1.0,
            #                           self.global_optimization_step,
            #                           threshold=500)

            loss = nll_loss + kl_loss + gan_loss * gan_weight * gan_factor
            log = { "loss:": loss.clone().detach(),
                    "nll loss:": nll_loss.clone().detach(),
                    # "l1_recon_loss:": l1_recon_loss.clone().detach(),
                    # "perceptual_recon_loss:": perceptual_recon_loss.clone().detach(),
                    "kl_loss:": kl_loss.clone().detach(),
                    "gan_loss:": gan_loss.clone().detach(),
                    "gan_weight:": gan_weight.clone().detach(),
            }
        elif optimizer_idx == 1:
            # update gan discriminator
            if cond is None:
                logits_fake = self.discriminator(recon_x.contiguous().detach())
                logits_real = self.discriminator(x.contiguous().detach())
            else:
                logits_fake = self.discriminator(
                    torch.cat((recon_x.contiguous().detach(), cond), dim=1))
                logits_real = self.discriminator(
                    torch.cat((x.contiguous().detach(), cond), dim=1))

            # gan_factor = adopt_weight(1.0,
            #                           self.global_optimization_step,
            #                           threshold=500)
            
            gan_loss = self.discriminator_loss(logits_real,
                                           logits_fake)
            
            # gan_loss = F.binary_cross_entropy_with_logits(logits_real, torch.ones_like(logits_real,
            #                                                                             device=logits_real.device))

            loss = gan_loss * gan_factor
            log = {
                "discriminator loss:": loss.clone().detach(),
                "gan loss:": gan_loss.clone().detach(), 
            }

        else:
            raise ValueError("optimzier idx out range")
        return loss, log

    def calculate_adaptive_weight(self,
                                  nll_loss,
                                  gan_loss,
                                  discriminator_weight=0.5):

        nll_grads = torch.autograd.grad(nll_loss,
                                        self.model.get_last_layer(),
                                        retain_graph=True)[0]
        gan_grads = torch.autograd.grad(gan_loss,
                                        self.model.get_last_layer(),
                                        retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(gan_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * discriminator_weight
        return d_weight
