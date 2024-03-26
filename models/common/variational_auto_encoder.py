import torch
from torch import nn
import torch.nn.functional as F
from torch import optim

import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.stats import norm
from functools import partial

from .conditional_unet import ConditionalUNet


class VariationalAutoEncoder(nn.Module):
    def __init__(self,image_size,image_chann=3,base_chann=64,output_chann=3,latent_chann=3):
        super().__init__()
        
        self.model = ConditionalUNet(image_size,image_size,image_chann,base_chann,
                                               output_channel=output_chann,short_cuts=False)

        self.encoded_image_h = self.model.encoded_image_h
        self.encoded_image_w = self.model.encoded_image_w
        self.encoded_image_c = self.model.encoded_image_c
        self.mean_linear = nn.Linear(self.encoded_image_c*self.encoded_image_h*self.encoded_image_w, 
                                     latent_chann)
        self.logvar_linear = nn.Linear(self.encoded_image_c*self.encoded_image_h*self.encoded_image_w, 
                                       latent_chann)
        self.decode_projection = nn.Linear(latent_chann, 
                                           self.encoded_image_c*self.encoded_image_h*self.encoded_image_w)
        self.out_sigmoid = nn.Sigmoid()

    def reparameter(self,mean,logvar):
        eps = torch.randn_like(mean)  #random sample from (0,1)
        return mean + torch.exp(logvar * 0.5) * eps  # u + sqrt(var)*eps ~ N(u, var)    
    
    def forward(self, x, t=None, x_self_cond=None):
        b,c,h,w = x.shape
        d = x.device
        
        encoded = self.model.encode(x,t,x_self_cond).view(b,-1)
        mean = self.mean_linear(encoded) #[b,latent_chann]
        logvar = self.logvar_linear(encoded) #[b,latent_chann]
        
        z = self.reparameter(mean, logvar) 
        
        recon_x = self.decode_projection(z).view(b,self.encoded_image_c,self.encoded_image_h,self.encoded_image_w)
        recon_x = self.model.decode(recon_x)
        recon_x = self.out_sigmoid(recon_x) #[b,c,h,w]
        return recon_x, mean, logvar

    def decode_latent(self, z):
        b,_ = z.shape
        recon_x = self.decode_projection(z).view(b,self.encoded_image_c,self.encoded_image_h,self.encoded_image_w)
        recon_x = self.model.decode(recon_x)
        recon_x = self.out_sigmoid(recon_x) #[b,c,h,w]
        return recon_x
    
    def get_kl_loss(self, mean, logvar):
        kl_loss = torch.mean(
             -0.5 * torch.sum(1+logvar-mean.pow(2)-torch.exp(logvar),1),0)

        return kl_loss
    
    def get_recon_loss(self, x, recon_x,):
        # recon_loss = torch.mean(torch.sum(F.binary_cross_entropy(recon_x,x,reduction='none'),
        #                                   dim=(1,2,3)))
        recon_loss = torch.mean(torch.sum(F.mse_loss(recon_x,x,reduction='none'),
                                          dim=(1,2,3)))
        return recon_loss   

 
        


