import torch
import torch.nn as nn
from einops import rearrange

import copy


class Mlp(nn.Module):
    def __init__(self, in_channels, hidden_channels=None, out_channels=None,
                 apply_act=True, drop_rate=0.0):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        if apply_act:
            self.act = nn.GELU()
        else:
            self.act = None
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(drop_rate)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
        
class PatchEmbed(nn.Module):
    """
    Args:
        img_h: Default:128
        img_w: Default:3600
        patch_size: Default:4
        in_channels: Default:3 
        out_channels: Default:96
        apply_norm: Default:False  
    """
    def __init__(self, img_resolution, patch_size=4, in_channels=4,
                 out_channels=96, apply_norm=False):
        super().__init__()
        self.img_resolution = img_resolution
        self.patch_resolution = [img_resolution[0] // patch_size, img_resolution[1] // patch_size]
        self.num_patches = self.patch_resolution[0] * self.patch_resolution[1]
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # [(img_h - kernel_size)/stride + 1, (img_w - kernel_size)/stride + 1] =>
        # [img_h/patch_size, img_w/patch_size]
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=patch_size, stride=patch_size)
        
        if apply_norm:
            self.norm = nn.LayerNorm(self.out_channels)
        else:
            self.norm = None

    def forward(self, x):   
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_resolution[0] and W == self.img_resolution[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x
    
class PatchMerging(nn.Module):
    """
    patch downsample

    Args:
    """
    def __init__(self, img_resolution, in_channels, apply_norm=True):
        super().__init__()
        self.img_resolution = img_resolution
        self.in_channels = in_channels
        self.reduction = nn.Linear(4*in_channels, 2*in_channels, bias=False)
        if apply_norm:
            self.norm = nn.LayerNorm(4 * in_channels)
        else:
            self.norm = None
        
    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.img_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

class PatchExpand(nn.Module):
    """_summary_
    [B, H*W, C] -> [B, 2H*2W, C//2]
    """
    def __init__(self, img_resolution, in_channels, apply_norm=True):
        super().__init__()
        self.img_resolution = img_resolution
        self.in_channels = in_channels
        self.expand = nn.Linear(in_channels, 2*in_channels, bias=False)
        if apply_norm:
            self.norm = nn.LayerNorm(self.in_channels//2)
        else:
            self.norm = None
    def forward(self, x):
        # x: [B, H*W, C_in]
        x = self.expand(x)  # [B, H*W, 2C_in]
        B, L, C = x.shape
        assert L==self.img_resolution[0]*self.img_resolution[1], "input feature wrong size"
        
        x = x.view(B,self.img_resolution[0],self.img_resolution[1],C) # [B, H, W, 2C_in]
        x = rearrange(x, 'b h w (p1 p2 c) -> b (h p1) (w p2) c', p1=2, p2=2, c=C//4) # [B, 2*H, 2*W, 2C_in//4]
        x = x.view(B,-1,C//4) # [B, 2H*2W, 2C_in//4]
        if self.norm is not None:
            x = self.norm(x) # [B, 2H*2W, C_in//2]
        return x
    
class PatchExpand4X(nn.Module):
    """_summary_
    [B, H*W, C] -> [B, 4H*4W, C]
    """
    def __init__(self, img_resolution, in_channels, apply_norm=True):
        super().__init__()
        self.img_resolution = img_resolution
        self.in_channels = in_channels
        self.expand = nn.Linear(in_channels, 16*in_channels, bias=False)
        if apply_norm:
            self.norm = nn.LayerNorm(self.in_channels)
        else:
            self.norm = None
    def forward(self, x):
        # x: [B, H*W, C_in]
        x = self.expand(x)  # [B, H*W, 16C_in]
        B, L, C = x.shape
        assert L==self.img_resolution[0]*self.img_resolution[1], "input feature wrong size"
        
        x = x.view(B,self.img_resolution[0],self.img_resolution[1],C) # [B, H, W, 16C_in]
        x = rearrange(x, 'b h w (p1 p2 c) -> b (h p1) (w p2) c', p1=4, p2=4, c=C//16) # [B, 4*H, 4*W, C_in]
        x.view(B,-1,C//16) # [B, 4H*4W, C_in]
        
        if self.norm is not None:
            x = self.norm(x) # [B, 4H*4W, C_in]
        return x

def window_partition(x, window_size):
        # x[B, H, W, C]
        # return: (num_windows*B, window_size, window_size, C)
        B,H,W,C = x.shape
        x = x.view(B, H//window_size, window_size, W//window_size, window_size, C)
        windows = x.permute(0,1,3,2,4,5).contiguous().view(-1,window_size,window_size,C)
        return windows

def window_reverse(windows, window_size, H, W):
        # Merge windows back to H*W image allocation
        # windows [num_windows*B,window_size,window_size,C]
        # return [B,H,W,C]
        num_windows = int(H/window_size)*(W/window_size)
        B = int(windows.shape[0] / num_windows)
        x = windows.view(B,int(H//window_size),int(W//window_size),window_size,window_size,-1)
        x = x.permute(0,1,3,2,4,5).contiguous().view(B,H,W,-1)
        return x

class WindowAttention(nn.Module):
    def __init__(self, in_channels, window_size, num_heads, qkv_bias, qk_scale, drop_rate):
        # super().__init__()
        # self.in_channels = in_channels
        # self.window_size = [window_size, window_size]
        # self.num_heads = num_heads
        # head_channels = in_channels // num_heads
        # self.scale = head_channels ** -0.5

        # self.relative_position_bias_table = nn.Parameter(torch.zeros((2*self.window_size[0]-1)*(2*self.window_size[1]-1), num_heads))
        # trunc_normal_(self.relative_position_bias_table, std=.02)
        # # relative pos in window
        # coords_h = torch.arange(self.window_size[0])
        # coords_w = torch.arange(self.window_size[1])
        # coords = torch.stack(torch.meshgrid([coords_h, coords_w])) #[2, Wh, Ww], thus [(0,i,j),(1,i,j)] is a pair
        # coords_flatten = torch.flatten(coords, 1) #[2, Wh*Ww], thus [(0,i),(1,i)] is a pair
        # #[2, Wh*Ww, Wh*Ww], (0,i,j) is the x offset of pixel i and j, (1,i,j) is the y offset of pixel i and j. 
        # #i, j is the pixel after window flatten
        # relative_coords = coords_flatten[:,:,None] - coords_flatten[:,None,:]
        # #[Wh*Ww, Wh*Ww, 2]. i, j is the pixel after window flatten:
        # # (i,j,0) is the x offset of pixel i and j, values ranges from (-(Wh-1),(Wh-1))
        # # (i,j,1) is the y offset of pixel i and j, values ranges from (-(Ww-1),(Ww-1))
        # relative_coords = relative_coords.permute(1,2,0).contiguous() 
        # relative_coords[:,:,0] += self.window_size[0]-1 #values ranges from (0, 2(Wh-1))
        # relative_coords[:,:,1] += self.window_size[1]-1 #values ranges from (0, 2(Ww-1))
        # relative_coords[:,:,0] *= 2*self.window_size[1]-1 #values ranges from (0, 2(Wh-1)*(2Ww-1))
        # relative_position_index = relative_coords.sum(-1) #[Wh*Ww, Wh*Ww], values range [0, (2Wh-1)*(2Ww-1)]
        # self.register_buffer("relative_position_index", relative_position_index)

        # self.qkv = nn.Linear(in_channels, 3*in_channels, bias=qkv_bias)
        # self.proj = nn.Linear(in_channels, in_channels)
        # self.proj_drop = nn.Dropout(drop_rate)
        # self.softmax = nn.Softmax(-1)
   
        super().__init__()
        self.in_channels = in_channels
        self.window_size = [window_size, window_size]  # Wh, Ww
        self.num_heads = num_heads
        head_dim = in_channels // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(in_channels, in_channels * 3, bias=qkv_bias)
        #self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(in_channels, in_channels)
        self.proj_drop = nn.Dropout(drop_rate)

        torch.nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x, mask=None):
        # x[num_windows*Batch, Wh*Ww, C]
        # mask[num_windows, Wh*Ww, Wh*Ww] , Wh/Ww window size
        
        B,N,C = x.shape
        qkv = self.qkv(x).reshape(B,N,3,self.num_heads,C//self.num_heads).permute(2,0,3,1,4)
        q,k,v = qkv[0],qkv[1],qkv[2] #[B,num_heads,Wh*Ww,C//num_heads]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1)) #[B,num_heads,Wh*Ww,Wh*Ww]
        
        # get any two pixel relative position bias, get [Wh*Ww,Wh*Ww,nh]
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0]*self.window_size[1], self.window_size[0]*self.window_size[1], -1) #[Wh*Ww,Wh*Ww,nh]
        relative_position_bias = relative_position_bias.permute(2,0,1).contiguous() #[nh, Wh*Ww,Wh*Ww]
        # add relative position bias to all windows
        attn = attn + relative_position_bias.unsqueeze(0) #[B,num_heads,Wh*Ww,Wh*Ww]
        
        if mask is not None:
            nW = mask.shape[0]
            #[batch,num_windows,num_heads,Wh*Ww,Wh*Ww]
            attn = attn.view(B // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            #[batch*num_windows,num_heads,Wh*Ww,Wh*Ww]
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        #[num_windows*batch, num_heads, Wh*Ww, C//num_heads] -> [num_windows*batch, Wh*Ww, C] 
        x = (attn @ v).transpose(1,2).reshape(B,N,C) #[num_windows*batch, Wh*Ww, C] 
    
        #[num_windows*batch, Wh*Ww, C] 
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
        
class SwinTransformerLayer(nn.Module):
    class DropPath(nn.Module):
        def __init__(self,drop_rate=0):
            super().__init__()
            self.drop_rate = drop_rate

        def forward(self,x):
            # x [B, H*W, C]
            if self.drop_rate==0. or not self.training:
                return x
            keep_prob=1-self.drop_rate
            shape=(x.shape[0],)+(1,)*(x.ndim-1)
            random_tensor=x.new_empty(shape).bernoulli_(keep_prob)
            if keep_prob>0.0:
                random_tensor.div_(keep_prob) #(x.shape[0],1,1)
            # drop-out on first dimension (ignore whole batch)
            return x*random_tensor
    
    
    def __init__(self, img_resolution, in_channels, num_heads, window_size,
                 mlp_ratio, qkv_bias, qk_scale, drop_rate,
                 drop_path_rate, apply_norm, shift_size):
        super().__init__()
        self.img_resolution = img_resolution
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if window_size >= img_resolution[0] or window_size >= img_resolution[1]:
            self.shift_size = 0
            self.window_size = min(img_resolution[0], img_resolution[1])
            
        self.attn = WindowAttention(in_channels, window_size, num_heads=num_heads,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    drop_rate=drop_rate)
            
        if apply_norm:
            self.norm1 = nn.LayerNorm(in_channels)
            self.norm2 = nn.LayerNorm(in_channels)
        else:
            self.norm1 = None
            self.norm2 = None
            
        self.drop_path = self.DropPath(drop_path_rate)
        mlp_hidden_channel = int(in_channels * mlp_ratio)
        self.mlp = Mlp(in_channels=in_channels, hidden_channels=mlp_hidden_channel,
                       out_channels=in_channels, apply_act=True, drop_rate=drop_rate)
        
        # SW-MSA
        if self.shift_size>0:
            #shift size is (window_size//2) for SW-MSA
            H, W = self.img_resolution[0], self.img_resolution[1]
            img_mask = torch.zeros((1,H,W,1))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            # img_mask will be labeled like: (cyclic shift)
            # [0., 0., 0., 0., 1., 1., 2., 2.],
            # [0., 0., 0., 0., 1., 1., 2., 2.],
            # [0., 0., 0., 0., 1., 1., 2., 2.],
            # [0., 0., 0., 0., 1., 1., 2., 2.],
            # [3., 3., 3., 3., 4., 4., 5., 5.],
            # [3., 3., 3., 3., 4., 4., 5., 5.],
            # [6., 6., 6., 6., 7., 7., 8., 8.],
            # [6., 6., 6., 6., 7., 7., 8., 8.]
            for h in h_slices:
                for w in w_slices:
                    img_mask[:,h,w,:] = cnt
                    cnt+=1
            #(num_windows*B,window_size,window_size,1)        
            mask_windows = window_partition(img_mask, self.window_size) 
            mask_windows = mask_windows.view(-1, self.window_size*self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            #(num_windows*B,window_size,window_size), attention mask of each window.
            #Unrelated parts will be -100, thus softmax generates 0
            attn_mask = attn_mask.masked_fill(attn_mask!=0, float(-100)).masked_fill(attn_mask==0,float(0))
        else:
            attn_mask = None
            
        self.register_buffer("attn_mask", attn_mask)    
            
    # def window_partition(self, x, window_size):
    #     # x[B, H, W, C]
    #     # return: (num_windows*B, window_size, window_size, C)
    #     B,H,W,C = x.shape
    #     x = x.view(B, H//window_size, window_size, W//window_size, window_size, C)
    #     windows = x.permute(0,1,3,2,4,5).contiguous().view(-1,window_size,window_size,C)
    #     return windows
       
    # def window_reverse(self, windows, window_size, H, W):
    #     # Merge windows back to H*W image allocation
    #     # windows [num_windows*B,window_size,window_size,C]
    #     # return [B,H,W,C]
    #     num_windows = int(H/window_size)*(W/window_size)
    #     B = int(windows.shape[0] / num_windows)
    #     x = windows.view(B,int(H//window_size),int(W//window_size),window_size,window_size,-1)
    #     x.permute(0,1,3,2,4,5).contiguous().view(B,H,W,-1)
    #     return x
        
    def forward(self, x):
        H, W = self.img_resolution[0], self.img_resolution[1]
        B, L, C = x.shape
        assert L==H*W, "input feature wrong size"
        
        shortcut = x
        x = self.norm1(x)  #layer norm on C
        x = x.view(B,H,W,C)
        
        #cyclic shift
        if self.shift_size>0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size,-self.shift_size),dims=(1,2))
        else:
            shifted_x = x

        #partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        
        #(B*num_windows, window_size*window_size, C)
        x_windows = x_windows.view(-1, self.window_size*self.window_size, C)

        #W-MSA/SW-MSA, return [B*num_windows, window_size*window_size, C]
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        #merge windows, return [B*num_windows, window_size*window_size, C]
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        shifted_x = window_reverse(attn_windows, self.window_size, H, W)
        
        #shift img features back in SW-MSA
        if self.shift_size>0:
            x = torch.roll(shifted_x,shifts=(self.shift_size,self.shift_size),dims=(1,2))
        else:
            x = shifted_x
        x = x.view(B,H*W,C)
        
        #ffn
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
            
class SwinTransformerEncoderBlock(nn.Module):
    """

    Args:
        nn (_type_): _description_
    """

    def __init__(self, img_resolution, in_channels, num_heads, depth, window_size,
                 mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_rate=0.0,
                 drop_path_rate=0.0, apply_norm=True, apply_downsample=True):
        super().__init__()
        
        self.img_resolution = img_resolution
        self.in_channels = in_channels

        # swim transformer
        # TODO, self.layers
        self.blocks = nn.ModuleList([
            SwinTransformerLayer(img_resolution, in_channels=in_channels,
                                 num_heads=num_heads, window_size=window_size,
                                 mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop_rate=drop_rate, drop_path_rate=drop_path_rate,
                                 apply_norm=apply_norm, shift_size = 0 if (i%2==0) else window_size // 2)
            for i in range(depth)])
        
        # patch merging
        if apply_downsample:
            self.downsample = PatchMerging(img_resolution, in_channels=in_channels)
        else:
            self.downsample = None
        
    def forward(self, x):
        for layer in self.blocks:
            x = layer(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

class SwinTransformerDecoderBlock(nn.Module):
    """

    Args:
        nn (_type_): _description_
    """

    def __init__(self, img_resolution, in_channels, num_heads, depth, window_size,
                 mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_rate=0.0,
                 drop_path_rate=0.0, apply_norm=True, apply_upsample=True):
        super().__init__()
        
        self.img_resolution = img_resolution
        self.in_channels = in_channels

        # swim transformer
        # TODO, self.layers
        self.blocks = nn.ModuleList([
            SwinTransformerLayer(img_resolution, in_channels=in_channels,
                                 num_heads=num_heads, window_size=window_size,
                                 mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop_rate=drop_rate, drop_path_rate=drop_path_rate,
                                 apply_norm=apply_norm, shift_size=0 if (i%2==0) else window_size // 2)
            for i in range(depth)])
        
        # patch merging
        if apply_upsample:
            self.upsample = PatchExpand(img_resolution, in_channels=in_channels, apply_norm=True)
        else:
            self.upsample = None
        
    def forward(self, x):
        for layer in self.blocks:
            x = layer(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x
    
class SwinTransformerUnet(nn.Module):
    PATCH_SIZE = 4
    EMBED_CHANNEL = 96
    DEPTHS = [2,2,2,2] #layers in each block of encoder
    NUM_HEADS = [3,6,12,24] #attn heads in each block
    WINDOW_SIZE = 7
    DROP_RATE = 0.0
    DROP_PATH_RATE = 0.1
    def __init__(self, H, W, IN_CHAN, NUM_CLASSES):
        super().__init__()
        
        self.H = H
        self.W = W
        img_h = H
        img_w = W
        num_classes = NUM_CLASSES
        in_channel = IN_CHAN
        
        # embed features
        self.patch_embed = PatchEmbed((img_h, img_w), 
                                      patch_size=SwinTransformerUnet.PATCH_SIZE,
                                      in_channels=in_channel,
                                      out_channels=SwinTransformerUnet.EMBED_CHANNEL,
                                      apply_norm=True)

        img_h = img_h // SwinTransformerUnet.PATCH_SIZE
        img_w = img_w // SwinTransformerUnet.PATCH_SIZE
        channels = SwinTransformerUnet.EMBED_CHANNEL
        
        # encoder
        # TODO, self.encoder
        self.layers = nn.ModuleList()
        for i in range(len(SwinTransformerUnet.DEPTHS)):
            block = SwinTransformerEncoderBlock((img_h,img_w),
                                                in_channels=channels,
                                                num_heads=SwinTransformerUnet.NUM_HEADS[i],
                                                depth=SwinTransformerUnet.DEPTHS[i],
                                                window_size=SwinTransformerUnet.WINDOW_SIZE,
                                                mlp_ratio=4,
                                                qkv_bias=True,
                                                qk_scale=None, 
                                                drop_rate=SwinTransformerUnet.DROP_RATE,
                                                drop_path_rate=SwinTransformerUnet.DROP_PATH_RATE,
                                                apply_norm=True,
                                                apply_downsample=True if (i < len(SwinTransformerUnet.DEPTHS)-1) else False)
            self.layers.append(block)
            if i < len(SwinTransformerUnet.DEPTHS)-1:
                img_h = img_h // 2
                img_w = img_w // 2
                channels = channels * 2
        # TODO, self.norm
        self.norm = nn.LayerNorm(channels)

        # decoder
        # TODO, self.layers_up
        self.layers_up = nn.ModuleList()
        # TODO, self.skip
        self.concat_back_dim = nn.ModuleList()
        for i in range(len(SwinTransformerUnet.DEPTHS)):
            concat_linear = nn.Linear(2*channels, channels) if i > 0 else nn.Identity()
            
            if i==0:
                block = PatchExpand((img_h,img_w), in_channels=channels, apply_norm=True)
            else:
                block = SwinTransformerDecoderBlock((img_h, img_w), in_channels=channels, 
                                                    num_heads=SwinTransformerUnet.NUM_HEADS[len(SwinTransformerUnet.DEPTHS)-1-i],
                                                    depth=SwinTransformerUnet.DEPTHS[len(SwinTransformerUnet.DEPTHS)-1-i],
                                                    window_size=SwinTransformerUnet.WINDOW_SIZE,
                                                    mlp_ratio=4,
                                                    qkv_bias=True,
                                                    qk_scale=None,
                                                    drop_rate=SwinTransformerUnet.DROP_RATE,
                                                    drop_path_rate=SwinTransformerUnet.DROP_PATH_RATE,
                                                    apply_norm=True,
                                                    apply_upsample=True if (i < len(SwinTransformerUnet.DEPTHS)-1) else False)
            self.layers_up.append(block)
            self.concat_back_dim.append(concat_linear)
            if i < len(SwinTransformerUnet.DEPTHS)-1:
                channels = channels // 2
                img_h = img_h * 2
                img_w = img_w * 2
        # TODO, self.norm_decode
        self.norm_up = nn.LayerNorm(channels)   
        # output
        self.up = PatchExpand4X((img_h, img_w), in_channels=channels, apply_norm=True) 
        self.output = nn.Conv2d(channels, num_classes, kernel_size=1, bias=False)                      
        img_h = img_h * 4
        img_w = img_w * 4
        channels = channels
        assert (img_h == H and img_w == W and channels == SwinTransformerUnet.EMBED_CHANNEL), "unet feature map wrong shape"
                                                         
    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        
        # x [B,C,H,W]
        x = self.patch_embed(x) #[B,H/4*W/4,96]

        x_downsample = []
        for blk in self.layers:
            x_downsample.append(x)
            x = blk(x)                           
        x = self.norm(x)
        
        for idx, layer in enumerate(self.layers_up):
            if idx == 0:
                x = layer(x)
            else:
                x = torch.cat([x, x_downsample[len(SwinTransformerUnet.DEPTHS)-1-idx]],-1) 
                x = self.concat_back_dim[idx](x)
                x = layer(x)
        x = self.norm_up(x)
                   
        B, _, _ = x.shape                  
        x = self.up(x)
        x = x.view(B, self.H, self.W, -1)
        x = x.permute(0,3,1,2)
        x = self.output(x) 
        
        return x
   
class SwinUnet(nn.Module):
    # 1.supports image width == height only
    # 2.image size % 7 == 0
    def __init__(self, H, W, IN_CHAN, NUM_CLASSES):
        super().__init__()
        self.swin_unet = SwinTransformerUnet(H,W,IN_CHAN,NUM_CLASSES)
    def forward(self, x):
        logits = self.swin_unet(x)
        return logits
    def load_from(self, pretrained_path):
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model"  not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]:v for k,v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = self.swin_unet.load_state_dict(pretrained_dict,strict=False)
                # print(msg)
                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            model_dict = self.swin_unet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3-int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k:v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
                        del full_dict[k]

            msg = self.swin_unet.load_state_dict(full_dict, strict=False)
            # print(msg)
        else:
            print("none pretrain") 
     
     
        
if __name__ == '__main__':        
    import torchvision
    from torchvision import transforms
    import os
    from torch.nn.modules.loss import CrossEntropyLoss
    
    from datasets import SynapseDataset, ImageNetDataset
    from utils import DiceLoss
   
    BATCHSIZE = 32
    IMAGE_SIZE = 224
    IMAGE_C = 3
    NUM_CLASSES=10
    NUM_EPOCHS = 10
    PRETRAINED_CKPT = os.path.abspath(os.path.join(os.path.abspath(__file__), 
                                                   '../../../data/swin_unet/swin_tiny_patch4_window7_224.pth'))
    
    list_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), 
                                                   '../../../data/Synapse/lists'))
    data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), 
                                                   '../../../data/Synapse/train_npz'))
    train_dataset = SynapseDataset(data_dir,list_dir,'train',IMAGE_SIZE,IMAGE_SIZE)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=BATCHSIZE, 
                                               shuffle=True)
    
    ViT = SwinUnet(H=IMAGE_SIZE, W=IMAGE_SIZE, IN_CHAN=IMAGE_C,NUM_CLASSES=NUM_CLASSES)
    #ViT.load_from(PRETRAINED_CKPT)
    net = ViT.cuda()

    optimizer = torch.optim.Adam(ViT.parameters(), lr=0.001)
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(NUM_CLASSES)
    
    total_step = len(train_loader)
    for epoch in range(NUM_EPOCHS):
        ViT.train()
        for i, sampled_batch in enumerate(train_loader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            outputs = ViT(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.4 * loss_ce + 0.6 * loss_dice

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, NUM_EPOCHS, loss.item()))
