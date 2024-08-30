import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.normalization import GroupNorm
import math
from einops import rearrange, reduce, repeat
from functools import partial

from .utils import RMSNorm
from typing import Optional, Tuple

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
    def __init__(self, channel_in, channel_out, groups=32, dropout=0.0):
        super().__init__()
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.norm = nn.GroupNorm(groups, channel_in, eps=1e-6, affine=True) 
        self.act = nn.SiLU()  #?how about relu
        self.dropout = nn.Dropout(dropout)
        self.conv = nn.Conv2d(channel_in, channel_out, 3, padding=1)
        
    def forward(self, x):
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.conv(x)
        
        # if scale_shift is not None:
        #     scale, shift = scale_shift
        #     x = x * (scale + 1.0) + shift
        
        return x


class ResBlock(nn.Module):
    def __init__(self, channel_in, channel_out, time_embedding_channel,
                 groups):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.SiLU(), nn.Linear(time_embedding_channel, channel_out))

        self.block1 = Block(channel_in, channel_out, groups=groups, dropout=0.0)
        self.block2 = Block(channel_out, channel_out, groups=groups, dropout=0.0)

        self.shortcut = nn.Sequential(
            nn.Conv2d(channel_in, channel_out, 1),
            nn.BatchNorm2d(channel_out)
        ) if channel_in != channel_out else nn.Identity()

    def forward(self, x, time_embedding=None, cond=None):
        h = self.block1(x)

        scale_shift = None
        if time_embedding is not None:
            time_embedding = self.mlp(time_embedding)
            time_embedding = time_embedding[:, :, None, None]
            # scale_shift = time_embedding.chunk(2, dim=1)
            h = h + time_embedding

        h = self.block2(h)   
        return h + self.shortcut(x)


class MultiHeadsAttention(nn.Module):
    def __init__(self, channel_in, heads=-1, channel_head=32, num_mem_kv=4, dropout=0.):
        super().__init__()
        self.scale = channel_head**-0.5
        self.heads = int(channel_in / channel_head) if heads==-1 else heads
        self.channel_head = channel_head
        hidden_dim = self.heads * self.channel_head

        self.qkv = nn.Conv2d(channel_in, hidden_dim * 3, 1,
                             bias=False)  #linear?
        # self.ffn = nn.Conv2d(hidden_dim, channel_in, 1)  #linear?
        self.ffn = nn.Sequential(nn.Conv2d(hidden_dim, channel_in, 1),
                                 RMSNorm(channel_in))
        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(dropout)
        
        self.norm = RMSNorm(channel_in)


    def forward(self, x, time_embedding=None, cond=None):
        b, c, h, w = x.shape

        x = self.norm(x)
        qkv = self.qkv(x).chunk(3, dim=1)
        #[b,heads,h*w,channel_head]
        q,k,v = map(lambda t: rearrange(t, 'b (h c) x y -> b h (x y) c', h=self.heads),qkv)
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  #[b,heads,h*w,h*w]

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        out = attn @ v
        out = rearrange(out, 'b h (x y) c -> b (h c) x y', x=h, y=w) #[b, heads*channel_head,h,w]
        out = self.ffn(out)  #[b,c,h,w]

        return out+x


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
        
    def forward(self, x, time_embedding=None, cond=None):
        b, c, h, w = x.shape
        x_clone = x.clone()
        
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
        return out+x_clone


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


class MultiHeadsCrossLinearAttention(nn.Module):
    def __init__(self, query_channel, context_channel=None, heads=8, channel_head=64, dropout=0.):
        super().__init__()
        hidden_dim = channel_head * heads
        context_channel = context_channel if context_channel is not None else query_channel

        self.scale = channel_head ** -0.5
        self.heads= heads
        self.to_q = nn.Linear(query_channel, hidden_dim, bias=False)
        self.to_k = nn.Linear(context_channel, hidden_dim, bias=False)
        self.to_v = nn.Linear(context_channel, hidden_dim, bias=False)
        
        self.ffn = nn.Sequential(nn.Conv2d(hidden_dim, query_channel, 1),
                                 RMSNorm(query_channel))
        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(dropout)
        self.norm = RMSNorm(query_channel)

    def forward(self, x, context=None, mask=None):
        b,c,h,w = x.shape
        x = self.norm(x)
        x_in = x.clone()
        
        if context is None:
            context = x
            context = rearrange(context, 'b c h w -> b (h w) c')
        elif context.dim()==2: #time embedding [b,c]
            context = context[:,:,None]
            context = rearrange(context, 'b c l -> b l c')
 
        x = rearrange(x, 'b c h w -> b (h w) c')

        q = self.to_q(x) #[b,n,channel_head*heads]
        k = self.to_k(context) #[b,l,channel_head*heads]
        v = self.to_v(context) #[b,l,channel_head*heads]

        #[b*heads,n,channel_head]
        q,k,v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=self.heads),(q,k,v))

        # q = q * self.scale
        # attn = (q @ k.transpose(-2, -1))  #[b*heads,n,n]
        
        q = q.softmax(dim=-1)
        k = k.softmax(dim=-2)
        q = q * self.scale
        attn = k.transpose(-2, -1) @ v  #[b*heads,n,l]


        if mask is not None:
            pass

 
        attn = self.attn_drop(attn)
        out = q @ attn
        out = rearrange(out, '(b h) n d -> b n (h d)', h=self.heads) #[b, n, heads*channel_head]
        out = rearrange(out, 'b (h w) c -> b c h w', h=h, w=w) #[b, heads*channel_head, h, w]
        
        out = self.ffn(out) #[b,c,h,w]

        return out + x_in


class MultiHeadsCrossAttentionBlock(nn.Module):
    def __init__(self, query_channel, output_channel=None, context_channel=None, heads=4, channel_head=32, dropout=0.):
        super().__init__()
        if output_channel == None:
            output_channel = query_channel
        #self atten
        self.attn1 = MultiHeadsCrossLinearAttention(query_channel=query_channel,context_channel=context_channel,
                                              heads=heads, channel_head=channel_head, dropout=dropout)
        #cross atten                                      
        self.attn2 = MultiHeadsCrossLinearAttention(query_channel=query_channel,context_channel=context_channel,
                                              heads=heads, channel_head=channel_head, dropout=dropout)
        #(todo):try gated gelu
        # self.ffn = nn.Sequential(
        #     nn.Linear(query_channel, query_channel*4),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(query_channel*4, query_channel)
        # )
        self.ffn = nn.Sequential(nn.Conv2d(query_channel, output_channel, 1),
                                 RMSNorm(output_channel))  

        self.norm1 = RMSNorm(query_channel)
        self.norm2 = RMSNorm(query_channel)
        self.norm3 = RMSNorm(query_channel)

        if query_channel != output_channel:
            self.conv_shortcut = nn.Conv2d(
                query_channel,
                output_channel,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
        else:
            self.conv_shortcut = nn.Identity()
    def forward(self, x, context=None):
        b,c,h,w = x.shape
        
        x_in = x.clone()
        #x = rearrange(x, 'b c h w -> b (h w) c', h=h,w=w)
        
        x = self.attn1(self.norm1(x)) 
        x = self.attn2(self.norm2(x), context=context) 
        x = self.ffn(self.norm3(x))
        
        #x = rearrange(x, 'b (h w) c -> b c h w', h=h,w=w)
        return x + self.conv_shortcut(x_in)


#############Implemented new version#############

class Upsample2D(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        norm_type=None,

    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels

        if norm_type == "ln_norm":
            self.norm = nn.LayerNorm(channels)
        elif norm_type == "rms_norm":
            self.norm = RMSNorm(channels)
        elif norm_type is None:
            self.norm = None
        else:
            raise ValueError(f"unknown norm_type: {norm_type}")

        
        self.conv = nn.Conv2d(self.channels, self.out_channels, kernel_size=3, padding=1, bias=True)

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        assert hidden_states.shape[1] == self.channels

        if self.norm is not None:
            hidden_states = self.norm(hidden_states.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
   
        hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
        hidden_states = self.conv(hidden_states)    

        return hidden_states


class Downsample2D(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        norm_type=None,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels

        if norm_type == "ln_norm":
            self.norm = nn.LayerNorm(channels, eps, elementwise_affine)
        elif norm_type is None:
            self.norm = None
        else:
            raise ValueError(f"unknown norm_type: {norm_type}")


        self.conv = nn.Conv2d(
            self.channels, self.out_channels, kernel_size=3, stride=2, padding=1, bias=True
        )
    

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        assert hidden_states.shape[1] == self.channels

        if self.norm is not None:
            hidden_states = self.norm(hidden_states.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        hidden_states = self.conv(hidden_states)

        return hidden_states


class ResBasicBlock2D(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        conv_shortcut: bool = False,
        temb_channels: int = 512,
        groups: int = 32,
        dropout=0.0,        
        time_embedding_norm: str = "default",  # default, scale_shift,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.time_embedding_norm = time_embedding_norm


        self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if temb_channels is not None:
            self.time_emb_proj = nn.Linear(temb_channels, out_channels) 
        else:
            self.time_emb_proj = None

        self.norm2 = torch.nn.GroupNorm(num_groups=groups, num_channels=out_channels, affine=True)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.nonlinearity = nn.SiLU()
        
        if self.in_channels != out_channels:
            self.conv_shortcut = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
        else:
            self.conv_shortcut = nn.Identity()

    def forward(self, input_tensor: torch.Tensor, temb: torch.Tensor, *args, **kwargs) -> torch.Tensor:

        hidden_states = input_tensor.clone()
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)

        if temb is not None and self.time_emb_proj is not None:
            temb = self.nonlinearity(temb)
            temb = self.time_emb_proj(temb)[:, :, None, None]

        if self.time_embedding_norm == "default":
            if temb is not None:
                hidden_states = hidden_states + temb
            hidden_states = self.norm2(hidden_states)
        elif self.time_embedding_norm == "scale_shift":
            if temb is None:
                raise ValueError(
                    f" `temb` should not be None when `time_embedding_norm` is {self.time_embedding_norm}"
                )
            time_scale, time_shift = torch.chunk(temb, 2, dim=1)
            hidden_states = self.norm2(hidden_states)
            hidden_states = hidden_states * (1 + time_scale) + time_shift
        else:
            hidden_states = self.norm2(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)
        output_tensor = (self.conv_shortcut(input_tensor) + hidden_states)

        return output_tensor


class ResDownBlock2D(nn.Module):
    def __init__(self, num_layers, in_channels, out_channels,
                  temb_channels=None, resnet_groups=32, dropout=0.0, add_downsample=False):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResBasicBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels 
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        out_channels, out_channels=out_channels
                    )
                ]
            )
        else:
            self.downsamplers = None      
         

    def forward(
        self, hidden_states: torch.Tensor, temb: Optional[torch.Tensor] = None, 
        context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        
        output_states = ()
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb)
            output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states = output_states + (hidden_states,)

        return hidden_states, output_states

class AttnDownBlock2D(nn.Module):
    def __init__(
        self,
        in_channels, out_channels, temb_channels=None, context_channels=None, num_layers=1, 
        resnet_groups=32, dropout=0.0, add_downsample=False, attention_head_dim=32):
        super().__init__()
        resnets = []
        attns = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResBasicBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels 
                )
            )
            attns.append(
                MultiHeadsCrossAttentionBlock(
                    query_channel = out_channels, 
                    context_channel=context_channels, 
                    heads=(out_channels//attention_head_dim), 
                    channel_head=attention_head_dim, 
                    dropout=0.
                )
            )
        self.resnets = nn.ModuleList(resnets)
        self.attentions = nn.ModuleList(attns)

        self.context_channels = context_channels
        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        out_channels, out_channels=out_channels
                    )
                ]
            )
        else:
            self.downsamplers = None      

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:

        output_states = ()
        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, temb)
            output_states = output_states + (hidden_states,)
            hidden_states = attn(hidden_states, context)
            output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states = output_states + (hidden_states,)

        return hidden_states, output_states

class AttnUpBlock2D(nn.Module):
    def __init__(self,
                in_channels, out_channels, prev_output_channel, temb_channels=None, 
                context_channels=None, num_layers=2,  resnet_groups=32, dropout=0.0, 
                add_upsample=False, add_down_up_skip=True, attention_head_dim=32):
        super().__init__()
        self.add_down_up_skip = add_down_up_skip
        layers = []

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels
            resnet_in_channels = resnet_in_channels + (res_skip_channels if self.add_down_up_skip else 0)

            if i==0:
                layers.append(
                    ResBasicBlock2D(
                        in_channels=resnet_in_channels,
                        out_channels=out_channels,
                        temb_channels=temb_channels
                    )
                )
            else:
                layers.append(
                    MultiHeadsCrossAttentionBlock(
                        query_channel = resnet_in_channels,
                        output_channel = out_channels,
                        context_channel=context_channels, 
                        heads=int(resnet_in_channels//attention_head_dim), 
                        channel_head=attention_head_dim, 
                        dropout=0.
                    )
                )

        self.layers = nn.ModuleList(layers)

        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, out_channels=out_channels)])
        else:
            self.upsamplers = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        res_hidden_states_tuple: Tuple[torch.Tensor, ...],
        temb: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            # pop res hidden states
            if self.add_down_up_skip:
                res_hidden_states = res_hidden_states_tuple[-1]
                res_hidden_states_tuple = res_hidden_states_tuple[:-1]
                hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            hidden_states = layer(hidden_states, context)
            
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states

class ResUpBlock2D(nn.Module):
    def __init__(self, num_layers,
                        in_channels,
                        out_channels,
                        prev_output_channel,
                        temb_channels,
                        add_down_up_skip=True,
                        add_upsample=True):
        super().__init__()
        self.add_down_up_skip = add_down_up_skip
        layers = []
        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels
            layers.append(
                ResBasicBlock2D(
                    in_channels=resnet_in_channels + (res_skip_channels if self.add_down_up_skip else 0),
                    out_channels=out_channels,
                    temb_channels=temb_channels
                )
            )

        self.layers = nn.ModuleList(layers)

        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, out_channels=out_channels)])
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False
        

    def forward(
        self,
        hidden_states: torch.Tensor,
        res_hidden_states_tuple: Tuple[torch.Tensor, ...],
        temb: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            # pop res hidden states
            if self.add_down_up_skip:
                res_hidden_states = res_hidden_states_tuple[-1]
                res_hidden_states_tuple = res_hidden_states_tuple[:-1]
                hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            hidden_states = layer(hidden_states, temb)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states

class ResMidBlock2D(nn.Module):
    def __init__(self, in_channels, temb_channels, num_layers=1):
        super().__init__()

        resnets = [
                ResBasicBlock2D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels
                )
        ]

        attentions = []
        for _ in range(num_layers):
                attentions.append(
                    # MultiHeadsCrossLinearAttention(
                    #     in_channels, temb_channels,
                    #     heads=in_channels // 8, 
                    #     channel_head=8
                    # )
                    AttnDownBlock2D(in_channels=in_channels, out_channels=in_channels, 
                                    temb_channels=temb_channels, context_channels=None, 
                                    num_layers=1, resnet_groups=32, dropout=0.0, 
                                    add_downsample=False, attention_head_dim=32)
                )
                resnets.append(
                    ResBasicBlock2D(
                        in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels)
                )
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(self, hidden_states, temb = None, context=None) -> torch.Tensor:
        hidden_states = self.resnets[0](hidden_states, temb)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                hidden_states = attn(hidden_states, temb)[0]
            hidden_states = resnet(hidden_states, temb)

        return hidden_states

class ConditionalUNet(nn.Module):
    def __init__(self,
                 image_h=32,
                 image_w=32,
                 image_c=3,
                 base_channel=64,
                 channel_mults=(1, 2, 4, 4),
                 output_channel=64,
                 resnet_block_groups=8, 
                 add_down_up_skip=True,
                 context_c=None):
        super().__init__()

        self.add_down_up_skip = add_down_up_skip
        self.input_channel = image_c
        self.output_channel = output_channel
        channels = [base_channel * m for m in channel_mults]
        time_channel = channels[0] * 4

        self.stem = nn.Conv2d(self.input_channel, channels[0], kernel_size=3, padding=(1, 1))

        self.time_mlp = nn.Sequential(
            PositionalEmbedding(channels[0]),
            nn.Linear(channels[0], time_channel),
            nn.SiLU(),
            nn.Linear(time_channel, time_channel),
        )

        self.downs = nn.ModuleList([])
        output_channel = channels[0]
        for i in range(len(channels)):
            is_last = i == (len(channels) - 1)
            input_channel = output_channel
            output_channel = channels[i]
            if not is_last:
                self.downs.append(
                        ResDownBlock2D(num_layers=2, in_channels=input_channel, out_channels=output_channel,
                                    temb_channels=time_channel, add_downsample=not is_last)
                )
            else:
                self.downs.append(
                        AttnDownBlock2D(in_channels=input_channel, out_channels=output_channel, 
                                        temb_channels=time_channel, context_channels=context_c, num_layers=1, 
                                        resnet_groups=32, dropout=0.0, add_downsample=not is_last, attention_head_dim=32)
                )

            if not is_last:
                image_h = image_h // 2
                image_w = image_w // 2
        self.encoded_image_h = image_h
        self.encoded_image_w = image_w    
        self.encoded_image_c = channels[-1]
        

        self.mids = nn.ModuleList([])
        self.mids.append(
                ResMidBlock2D(in_channels=channels[-1],
                               temb_channels=time_channel)
        )

        self.ups = nn.ModuleList([])
        reversed_channels = list(reversed(channels))
        output_channel = reversed_channels[0]
        for i in range(len(channels)):
            is_last = i == len(channels) - 1
            prev_output_channel = output_channel
            output_channel = reversed_channels[i]
            input_channel = reversed_channels[min(i + 1, len(channels) - 1)]
            self.ups.append(
                    ResUpBlock2D(num_layers=3,
                            in_channels=input_channel,
                            out_channels=output_channel,
                            prev_output_channel=prev_output_channel,
                            temb_channels=time_channel,
                            add_down_up_skip=add_down_up_skip,
                            add_upsample=not is_last)
            )

            # self.ups.append(
            #         AttnUpBlock2D(input_channel, output_channel, prev_output_channel,
            #                      temb_channels=time_channel, context_channels=None, num_layers=3, 
            #                       resnet_groups=32, dropout=0.0, add_upsample=not is_last, 
            #                       add_down_up_skip=add_down_up_skip, attention_head_dim=32)
            # )

            if not is_last:
                image_h = image_h * 2
                image_w = image_w * 2

    
        self.head = nn.Sequential(
                nn.GroupNorm(num_channels=reversed_channels[-1], num_groups=resnet_block_groups),
                nn.SiLU(),
                nn.Conv2d(reversed_channels[-1], self.output_channel, kernel_size=3, padding=1)
        )

    def get_last_layer(self):
        return self.head[-1].weight

    def forward(self, x, time=None, cond=None):
        # x[b,c,h,w]
        # time[b,]
        # cond[b,c]

        x = self.stem(x)  #[b,64,h,w]
        t = self.time_mlp(time) if time is not None else None  #[b,256]

        shortcut = (x,)
        for block in self.downs:
            x, res = block(x, t, cond)
            shortcut += res

        for block in self.mids:
            x = block(x, t, cond)

        for block in self.ups:
            if self.add_down_up_skip:
                res_samples = shortcut[-len(block.layers) :]
                shortcut = shortcut[: -len(block.layers)]
            else:
                res_samples = None
            x = block(x, res_samples, t, cond)

        #x = torch.cat((x, r), dim=1)
        for block in self.head:
            x = block(x)

        return x

    def encode(self,x,time=None,cond=None):
        # x[b,c,h,w]
        # time[b,]
        # cond[b,c]

        x = self.stem(x)  #[b,64,h,w]
        t = self.time_mlp(time) if time is not None else None  #[b,256]

        shortcut = (x,)
        for block in self.downs:
            x, res = block(x, t, cond)
            shortcut += res

        for block in self.mids:
            x = block(x, t, cond)

        return x

    def decode(self,x,time=None,cond=None):
        t = self.time_mlp(time) if time is not None else None  #[b,256]

        for block in self.mids:
            x = block(x, t, cond)

        for block in self.ups:
            x = block(x, None, t, cond)

        for block in self.head:
            x = block(x)
        return x


