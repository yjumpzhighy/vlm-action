import torch
import torch.nn as nn
from scipy import ndimage
import numpy as np
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes
    
    
def inception_score(logits, eps=1e-9):
    #logits [N,C]
    assert len(logits.shape) == 2, f"Tensor has wrong dimension"
    
    p_yx = F.softmax(logits, dim=-1)
    #Marginalize class probabilities to get class distribution
    p_y = torch.mean(p_yx, dim=0, keepdim=True) #[1,C]
    kl = p_yx * (torch.log(p_yx+eps) - torch.log(p_y+eps)) #[B,N,C]
    score = torch.exp(kl.sum(axis=1).mean())
    return score
   
def frechet_inception_distance(feat_f, feat_t):
    #feat_f, fake image features [N,C]
    #feat_g, true image features [N,C]
    assert len(feat_f.shape) == 2, f"Tensor has wrong dimension"
    assert len(feat_t.shape) == 2, f"Tensor has wrong dimension"
    
    # mean and var on each channel
    mean_f = torch.mean(feat_f, axis=0) #[C,]
    mean_t = torch.mean(feat_t, axis=0)
    
    var_f = torch.cov(feat_f.T) #[C,C]
    var_t = torch.cov(feat_t.T) 

    fid = torch.sum((mean_f - mean_t)**2) + torch.trace(var_f + var_t - 2 * torch.sqrt(torch.mul(var_f, var_t)))
    return fid
    
class EMA(object):
    # v(t) = decay*v(t-1) + (1-decay)*theta(t)
    # v is ema weights, theta is model true weights
    
    '''
    ema = EMA(model, 0.999)
    ema.register()
    # train
    optimizer.step()
    ema.update()
    # evaluate
    ema.apply_shadow()
    ema.restore()
    '''
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
