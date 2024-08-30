import torch
from torch import nn
import torch.nn.functional as F
from torch import optim

import matplotlib.pyplot as plt
import os
import sys
import numpy as np
from scipy.stats import norm
from functools import partial
from taming.modules.losses.vqperceptual import *

from .conditional_unet import ConditionalUNet






class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=[1, 2, 3])
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3])

    def nll(self, sample, dims=[1,2,3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean

class LPIPSWithDiscriminator(nn.Module):
    def __init__(self, disc_start=0, logvar_init=0.0, kl_weight=0.000001, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=0.5,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_loss="hinge"):

        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm
                                                 ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, inputs, reconstructions, posteriors, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train",
                weights=None):
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss

        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights*nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
            g_loss = -torch.mean(logits_fake)

            if self.disc_factor > 0.0:
                try:
                    d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                except RuntimeError:
                    assert not self.training
                    d_weight = torch.tensor(0.0)
            else:
                d_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = weighted_nll_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(), "{}/logvar".format(split): self.logvar.detach(),
                   "{}/kl_loss".format(split): kl_loss.detach().mean(), "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   }
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            return d_loss, log

    

class VariationalAutoEncoder(nn.Module):
    def __init__(self,
                 model,
                 image_size,
                 image_chann=3,
                 output_chann=3,
                 latent_chann=3,
                 use_GAN=False,
                 num_training_steps=None):
        super().__init__()

        self.model = model

        self.encoded_image_h = self.model.encoded_image_h
        self.encoded_image_w = self.model.encoded_image_w
        self.encoded_image_c = self.model.encoded_image_c
        
        self.latent_image_h = self.model.encoded_image_h
        self.latent_image_w = self.model.encoded_image_w
        self.latent_chann = latent_chann
        
        self.mean_logvar = nn.Conv2d(self.encoded_image_c, 2*latent_chann, 1)
        self.decode_projection = nn.Conv2d(latent_chann, self.encoded_image_c, 1)

        self.use_GAN = use_GAN
        if use_GAN:
            self.discriminator = NLayerDiscriminator(
                    input_nc=output_chann, n_layers=3,
                    use_actnorm=False).apply(weights_init)
        else:
            self.discriminator = None
        self.discriminator_loss = hinge_d_loss
        self.discriminator_bce_loss = nn.BCEWithLogitsLoss()

        self.perceptual_loss = LPIPS().eval() #fronzon
        self.recon_loss_regulizer = nn.Parameter(torch.zeros(size=()))

        self.loss_func = LPIPSWithDiscriminator(disc_start=int(num_training_steps*0.6))

    def reparameter(self, mean, logvar):
        eps = torch.randn_like(mean)  #random sample from (0,1)
        return mean + torch.exp(
            logvar * 0.5) * eps  # u + sqrt(var)*eps ~ N(u, var)

    def forward(self, x, t=None, cond=None, mode='full', optimizer_idx=0, gan_factor=0.0, global_step=0): 
        if mode == 'decode':
            return self.decode(x)

        if mode == 'encode':
            return self.encode(x,t,cond)

        b, c, h, w = x.shape
        d = x.device

        posterior = self.encode(x, t, cond)
        z = posterior.sample()
        recon_x = self.decode(z)
        

        loss = self.get_loss(posterior, x, recon_x, optimizer_idx=optimizer_idx, 
                                  gan_factor=gan_factor, global_step=global_step)

        return loss

    def decode(self, z):
        #b, _ = z.shape
        recon_x = self.decode_projection(z)
        recon_x = self.model.decode(recon_x)
        return recon_x

    def encode(self, x, t=None, cond=None):
        
        encoded = self.model.encode(x, t, cond) #[b,encoded_chann,h',w']
        mean_logvar = self.mean_logvar(encoded) #[b,2*encoded_chann,h',w']

        # mean, logvar = torch.chunk(mean_logvar, 2, dim=1) 
        # z = self.reparameter(mean, logvar) #[b,latent_chann,h',w']
        posterior = DiagonalGaussianDistribution(mean_logvar)

        return posterior

    def get_kl_loss(self, mean, logvar):
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - torch.exp(logvar), dim=[1,2,3])

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
                self.loss_func.discriminator.parameters(), lr=lr, betas=(0.5, 0.9))
        else:
            optim_discriminator = None

        return [optim_autoencoder, optim_discriminator]

    def get_loss(self, posterior, x, recon_x, optimizer_idx, gan_factor=0.0, global_step=0, cond=None):
        aeloss, log = (self.loss_func(x,recon_x,posterior,optimizer_idx, global_step,
                          last_layer=self.get_last_layer(), split="train"))

        loss = (aeloss,)
        return loss

    def get_last_layer(self):
        return self.model.get_last_layer()

    def calculate_adaptive_weight(self,
                                  nll_loss,
                                  gan_loss,
                                  discriminator_weight=0.5):

        nll_grads = torch.autograd.grad(nll_loss,
                                        self.get_last_layer(),
                                        retain_graph=True)[0]
        gan_grads = torch.autograd.grad(gan_loss,
                                        self.get_last_layer(),
                                        retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(gan_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * discriminator_weight
        return d_weight




