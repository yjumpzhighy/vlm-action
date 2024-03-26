import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.normalization import GroupNorm
import math
from einops import rearrange, reduce, repeat
from functools import partial

from .utils import RMSNorm

class PositionalEmbedding(nn.Module):
    def __init__(self, dmodel, scale=1.0):
        super().__init__()
        self.dmodel = dmodel  #embedding dimension
        self.scale = scale

    def forward(self, time):
        #sin(t / 10000^(i/(dmodel//2))), cos(t / 10000^(i/(dmodel//2))), i~[0, dmodel//2]
        device = time.device
        half_dmodel = self.dmodel // 2
        embeddings = math.log(10000) / (half_dmodel - 1)
        embeddings = torch.exp(
            torch.arange(half_dmodel, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class NormalizedConv2d(nn.Conv2d):
    '''
    replace wide resnet
    '''
    def forward(self, x):
        eps = 1e-5
        weight = self.weight
        # [C_out,C_in,k,k] get mean/var along first dimension [C_out,1,1,1]
        mean = torch.mean(weight, dim=(1, 2, 3), keepdim=True)
        var = torch.var(weight, dim=(1, 2, 3), unbiased=False, keepdim=True)
        # making weight distribution be N(0, I)
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(x, normalized_weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class Block(nn.Module):
    def __init__(self, channel_in, channel_out, groups):
        super().__init__()
        self.channel_in = channel_in
        self.channel_out = channel_out
        #self.conv = NormalizedConv2d(channel_in, channel_out, 3, padding=1)
        self.conv = nn.Conv2d(channel_in, channel_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, channel_out)  #?how about batch norm?
        #self.norm = nn.BatchNorm2d(channel_out)
        self.act = nn.SiLU()  #?how about relu

    def forward(self, x, scale_shift=None):
        x = self.conv(x)
        x = self.norm(x)
        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1.0) + shift
        x = self.act(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, channel_in, channel_out, time_embedding_channel,
                 groups):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.SiLU(), nn.Linear(time_embedding_channel, channel_out * 2))

        self.block1 = Block(channel_in, channel_out, groups=groups)
        self.block2 = Block(channel_out, channel_out, groups=groups)

        self.shortcut = nn.Sequential(
            nn.Conv2d(channel_in, channel_out, 1),
            nn.BatchNorm2d(channel_out)
        ) if channel_in != channel_out else nn.Identity()

    def forward(self, x, time_embedding=None):
        scale_shift = None
        if time_embedding is not None:
            time_embedding = self.mlp(time_embedding)
            time_embedding = time_embedding[:, :, None, None]
            scale_shift = time_embedding.chunk(2, dim=1)

        h = self.block1(x, scale_shift)
        h = self.block2(h)
        return h + self.shortcut(x)


class MultiHeadsAttention(nn.Module):
    def __init__(self, channel_in, heads=4, channel_head=32, num_mem_kv=4, dropout=0.):
        super().__init__()
        self.scale = channel_head**-0.5
        self.heads = heads
        self.channel_head = channel_head
        hidden_dim = heads * channel_head
        
        self.qkv = nn.Conv2d(channel_in, hidden_dim * 3, 1,
                             bias=False)  #linear?
        # self.ffn = nn.Conv2d(hidden_dim, channel_in, 1)  #linear?
        self.ffn = nn.Sequential(nn.Conv2d(hidden_dim, channel_in, 1),
                                 RMSNorm(channel_in))
        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(dropout)
        
        self.norm = RMSNorm(channel_in)
        self.mem_kv = nn.Parameter(torch.randn(2,heads,num_mem_kv,channel_head))

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)
        qkv = self.qkv(x).chunk(3, dim=1)
        #[b,heads,h*w,channel_head]
        q,k,v = map(lambda t: rearrange(t, 'b (h c) x y -> b h (x y) c', h=self.heads),qkv)
        
        # #[b,heads,num_mem_kv,channel_head]
        # mk,mv = map(lambda t: repeat(t, 'h n c -> b h n c', b=b), self.mem_kv)
        # k,v = map(partial(torch.cat, dim=-2), ((mk,k),(mv,v)))
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  #[b,heads,h*w,h*w]

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        out = attn @ v
        out = rearrange(out, 'b h (x y) c -> b (h c) x y', x=h, y=w) #[b, heads*channel_head,h,w]
        out = self.ffn(out)  #[b,c,h,w]

        return out


class MultiHeadsLinearAttention(nn.Module):
    def __init__(self, channel_in, heads=4, channel_head=32, num_mem_kv=4, dropout=0.):
        super().__init__()
        self.scale = channel_head**-0.5
        self.heads = heads
        self.channel_head = channel_head
        hidden_dim = heads * channel_head
        
        self.qkv = nn.Conv2d(channel_in, hidden_dim * 3, 1,
                             bias=False)  #linear?
        self.ffn = nn.Sequential(nn.Conv2d(hidden_dim, channel_in, 1),
                                 RMSNorm(channel_in))  
        self.norm = RMSNorm(channel_in)
        self.mem_kv = nn.Parameter(torch.randn(2,heads,num_mem_kv,channel_head))
        
    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)
        qkv = self.qkv(x).chunk(3, dim=1)
        #[b,heads,h*w,channel_head]
        q,k,v = map(lambda t: rearrange(t, 'b (h c) x y -> b h (x y) c', h=self.heads),qkv)

        # #[b,heads,num_mem_kv,channel_head]
        # mk,mv = map(lambda t: repeat(t, 'h n c -> b h n c', b=b), self.mem_kv)
        # k,v = map(partial(torch.cat, dim=-2), ((mk,k),(mv,v)))
        
        q = q.softmax(dim=-1)
        k = k.softmax(dim=-2)
        q = q * self.scale
        attn = k.transpose(-2, -1) @ v  #[b,heads,channel_head,channlel_head]
        out = q @ attn #[b,heads,h*w,channlel_head]
        out = rearrange(out, 'b h (x y) c -> b (h c) x y', x=h, y=w) #[b, heads*channel_head,h,w]
        out = self.ffn(out)  #[b,c,h,w]
        return out


class Downsample(nn.Module):
    '''
    downsample H,W to H//2,W//2
    '''
    def __init__(self, channels_in, channels_out):
        super().__init__()
        #self.downsample = nn.Conv2d(channels_in, channels_in, 3, stride = 2, padding=1)

        self.downsample = nn.Conv2d(channels_in * 4, channels_out, 1)

    def forward(self, x):
        
        # x:(b,c,h,w)
        b, c, h, w = x.shape

        if h % 2 == 1 or w % 2 == 1:
            raise ValueError("downsampling dimension must be even")

        x = x.view(b, c, h // 2, 2, w // 2,
                   2).permute(0, 1, 3, 5, 2,
                              4).contiguous().view(b, c * 4, h // 2, w // 2)
        return self.downsample(x)


class Upsample(nn.Module):
    '''
    upsample H,W to 2H,2W
    '''
    def __init__(self, channel_in, channel_out):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(channel_in, channel_out, 3, padding=1))

    def forward(self, x):
        return self.upsample(x)


class ConditionalUNet(nn.Module):
    def __init__(self,
                 image_h=32,
                 image_w=32,
                 image_c=3,
                 base_channel=64,
                 channel_mults=(1, 2, 4, 8),
                 output_channel=64,
                 self_condition=False,
                 resnet_block_groups=8,
                 short_cuts=True):
        super().__init__()

        self.self_condition = self_condition
        input_channel = image_c * (2 if self_condition else 1)
        self.short_cuts = short_cuts

        self.stem = nn.Conv2d(input_channel, base_channel, 1, padding=0)

        channels = [base_channel] + [base_channel * m for m in channel_mults]
        channels_in = channels[:-1]
        channels_out = channels[1:]

        time_channel = base_channel * 4
        self.time_mlp = nn.Sequential(
            PositionalEmbedding(base_channel),
            nn.Linear(base_channel, time_channel),
            nn.GELU(),
            nn.Linear(time_channel, time_channel),
        )

        self.downs = nn.ModuleList([])

        for ind in range(len(channels_in)):
            is_last = ind == (len(channels_in) - 1)
            self.downs.append(
                nn.ModuleList([
                    ResBlock(channels_in[ind], channels_in[ind], time_channel,
                             resnet_block_groups),
                    ResBlock(channels_in[ind], channels_in[ind], time_channel,
                             resnet_block_groups),
                    MultiHeadsLinearAttention(channels_in[ind]),
                    Downsample(channels_in[ind], channels_out[ind])
                    if not is_last else nn.Conv2d(
                        channels_in[ind], channels_out[ind], 3, padding=1)
                ]))
            if not is_last:
                image_h = image_h // 2
                image_w = image_w // 2
        self.encoded_image_h = image_h
        self.encoded_image_w = image_w    
        self.encoded_image_c = channels_out[-1]
        

        self.mids = nn.ModuleList([])
        self.mids.append(
            nn.ModuleList([
                ResBlock(channels_out[-1], channels_out[-1], time_channel,
                         resnet_block_groups),
                MultiHeadsLinearAttention(channels_out[-1]),
                ResBlock(channels_out[-1], channels_out[-1], time_channel,
                         resnet_block_groups),
            ]))

        self.ups = nn.ModuleList([])
        for ind in reversed(range(len(channels_out))):
            is_last = ind == 0
            self.ups.append(
                nn.ModuleList([
                    ResBlock(channels_out[ind] + channels_in[ind] if short_cuts else channels_out[ind],
                             channels_out[ind], time_channel,
                             resnet_block_groups),
                    ResBlock(channels_out[ind] + channels_in[ind] if short_cuts else channels_out[ind],
                             channels_out[ind], time_channel,
                             resnet_block_groups),
                    MultiHeadsLinearAttention(channels_out[ind]),
                    Upsample(channels_out[ind], channels_in[ind])
                    if not is_last else nn.Conv2d(
                        channels_out[ind], channels_in[ind], 3, padding=1)
                ]))
            if not is_last:
                image_h = image_h * 2
                image_w = image_w * 2

        self.head = nn.ModuleList([])
        self.head.append(
            nn.ModuleList([
                ResBlock(channels_in[0] + base_channel if short_cuts else channels_in[0],
                         base_channel,
                         time_channel, resnet_block_groups),
                nn.Conv2d(base_channel, output_channel, 1)
            ]))

        self.classifier = nn.ModuleList([])
        self.classifier.append(
            nn.ModuleList([
                nn.Flatten(),
                nn.Linear(image_h * image_w * output_channel, output_channel),
                nn.Linear(output_channel, output_channel)
            ]))

    def forward(self, x, time=None, x_self_cond=None):
        # x[b,3,h,w]
        # time[b,]
        assert self.short_cuts, f"unet encode-decode process requires shortcuts"
        if self.self_condition:
            if x_self_cond is None:
                x_self_cond = torch.zeros_like(x)
            x = torch.cat((x_self_cond, x), dim=1)  #[b,2*3,h,w]

        x = self.stem(x)  #[b,64,h,w]
        r = x.clone()

        t = self.time_mlp(time) if time is not None else None  #[b,256]

        shortcut = []
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            shortcut.append(x)

            x = block2(x, t)
            x = attn(x) + x
            shortcut.append(x)

            x = downsample(x)

        for block1, attn, block2 in self.mids:
            x = block1(x, t)
            x = attn(x) + x
            x = block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, shortcut.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, shortcut.pop()), dim=1)
            x = block2(x, t)
            x = attn(x) + x

            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        for block, proj in self.head:
            x = block(x, t)
            x = proj(x)

        return x

    def encode(self,x,time=None,x_self_cond=None):
        # x[b,3,h,w]
        # time[b,]

        if self.self_condition:
            if x_self_cond is None:
                x_self_cond = torch.zeros_like(x)
            x = torch.cat((x_self_cond, x), dim=1)  #[b,2*3,h,w]

        x = self.stem(x)  #[b,64,h,w]

        t = self.time_mlp(time) if time is not None else None  #[b,256]

        #shortcut = []
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            #shortcut.append(x)

            x = block2(x, t)
            #x = attn(x)
            #shortcut.append(x)

            x = downsample(x)


        for block1, attn, block2 in self.mids:
            x = block1(x, t)
            #x = attn(x)
            x = block2(x, t)
            
        return x #[b,h/8,w/8,512]
  
    def decode(self, x, time=None):
        assert not self.short_cuts, f"unet short cuts not supported in decode-only process"
        t = self.time_mlp(time) if time is not None else None  #[b,256]
        for block1, block2, attn, upsample in self.ups:
            x = block1(x, t)
            x = block2(x, t)
            #x = attn(x)

            x = upsample(x)

        for block, proj in self.head:
            x = block(x, t)
            x = proj(x)

        return x
            
    def classify_forward(self, x):
        x = self.forward(x)
        for flat, ln1, ln2 in self.classifier:
            x = ln2(ln1(flat(x)))
        return x


# ConditionalUNet unit test
if __name__ == "__main__":
    import torchvision
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transforms.ToTensor(),
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='./data/cifar',
                                                 train=True,
                                                 transform=transform,
                                                 download=False)

    # 测试数据集
    test_dataset = torchvision.datasets.CIFAR10(root='./data/cifar',
                                                train=False,
                                                transform=transform,
                                                download=False)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=32,
                                               shuffle=True)
    # 测试数据加载器
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=32,
                                              shuffle=False)

    net = ConditionalUNet(32,
                          32,
                          3,
                          base_channel=64,
                          output_channel=10,
                          self_condition=False).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    total_step = len(train_loader)
    num_epochs = 60
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.cuda()
            labels = labels.cuda()
            outputs = net.classify_forward(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  
        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                    epoch + 1, num_epochs, i + 1, total_step, loss.item()))

        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.cuda()
                labels = labels.cuda()
                outputs = net.classify_forward(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print('Test Accuracy of the model on the test images: {} %'.format(
                100 * correct / total))
