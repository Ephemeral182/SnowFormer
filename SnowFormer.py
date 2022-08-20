import torch
import torch.nn as nn
import numpy as np
from timm.models.layers import DropPath,to_2tuple,trunc_normal_
import math 
import time
from base_net_snow import *
import torch.nn.functional as F

def _to_channel_last(x):
    """
    Args:
        x: (B, C, H, W)
    Returns:
        x: (B, H, W, C)
    """
    return x.permute(0, 2, 3, 1)


def _to_channel_first(x):
    """
    Args:
        x: (B, H, W, C)
    Returns:
        x: (B, C, H, W)
    """
    return x.permute(0, 3, 1, 2)


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size: window size
    Returns:
        local window features (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: local window features (num_windows*B, window_size, window_size, C)
        window_size: Window size
        H: Height of image
        W: Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):

        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class channel_shuffle(nn.Module):
    def __init__(self,groups=3):
        super(channel_shuffle,self).__init__()
        self.groups = groups
    
    def forward(self,x):
        B,C,H,W = x.shape
        assert C % self.groups == 0
        C_per_group = C // self.groups
        x = x.view(B,self.groups,C_per_group,H,W)
        x = x.transpose(1,2).contiguous()

        x = x.view(B,C,H,W)
        return x

class overlapPatchEmbed(nn.Module):
    def __init__(self,img_size=224,patch_size=7,stride=4,in_channels=3,dim=768):
        super(overlapPatchEmbed,self).__init__()
        
        patch_size=to_2tuple(patch_size)

        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels,dim,kernel_size=patch_size,stride=stride,padding=(patch_size[0]//2,patch_size[1]//2))
        self.norm = nn.LayerNorm(dim)
        
        self.apply(self._init_weight)
    
    def _init_weight(self,m):
        if isinstance(m,nn.Linear):
            trunc_normal_(m.weight,std=0.02)
            if isinstance(m,nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias,0)
        elif isinstance(m,nn.LayerNorm):
            nn.init.constant_(m.bias,0)
            nn.init.constant_(m.weight,1.0)
        elif isinstance(m,nn.Conv2d):
            fan_out = m.kernel_size[0]*m.kernel_size[1]*m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0,math.sqrt(2.0/fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    
    def forward(self,x):
        x = self.proj(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_head=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_head == 0, f"dim {dim} should be divided by num_heads {num_head}."

        self.dim = dim
        self.num_heads = num_head
        head_dim = dim // num_head
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim,1,1)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv2d(dim,dim,3,1,1,groups=dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        
        B, N, C = x.shape
        x_conv = self.conv(x.reshape(B,H,W,C).permute(0,3,1,2))

        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x.transpose(1,2).reshape(B,C,H,W))
        x = self.proj_drop(x)
        x = x+x_conv
        return x


class SimpleGate(nn.Module):
    def forward(self,x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)


class MFFN(nn.Module):
    def __init__(self, dim, FFN_expand=2,norm_layer='WithBias'):
        super(MFFN, self).__init__()

        self.conv1 = nn.Conv2d(dim,dim*FFN_expand,1)
        self.conv33 = nn.Conv2d(dim*FFN_expand,dim*FFN_expand,3,1,1,groups=dim*FFN_expand)
        self.conv55 = nn.Conv2d(dim*FFN_expand,dim*FFN_expand,5,1,2,groups=dim*FFN_expand)
        self.sg = SimpleGate()
        self.conv4 = nn.Conv2d(dim,dim,1)

        self.apply(self._init_weights)
    def _init_weights(self,m):
        if isinstance(m,nn.Linear):
            trunc_normal_(m.weight,std=0.02)
            if isinstance(m,nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias,0)
        elif isinstance(m,nn.LayerNorm):
            nn.init.constant_(m.bias,0)
            nn.init.constant_(m.weight,1.0)
        elif isinstance(m,nn.Conv2d):
            fan_out = m.kernel_size[0]*m.kernel_size[1]*m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0,math.sqrt(2.0/fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    def forward(self, x):
        x1 = self.conv1(x)
        x33 = self.conv33(x1)
        x55 = self.conv55(x1)
        x = x1+x33+x55
        x = self.sg(x)
        x = self.conv4(x)
        return x

class Scale_aware_Query(nn.Module):
    def __init__(self,
                 dim,
                 out_channel,
                 window_size,
                 num_heads):
        super().__init__()
        self.dim = dim
        self.out_channel = out_channel
        self.window_size = window_size
        self.conv = nn.Conv2d(dim,out_channel,1,1,0)

        layers=[]
        for i in range(3):
            layers.append(CALayer(out_channel,4))
            layers.append(SALayer(out_channel,4))
        self.globalgen = nn.Sequential(*layers)

        self.num_heads = num_heads
        self.N = window_size * window_size
        self.dim_head = out_channel // self.num_heads

    def forward(self, x):
        x = self.conv(x)
        x = F.upsample(x,(self.window_size,self.window_size),mode="bicubic")
        x = self.globalgen(x)
        B = x.shape[0]
        x = x.reshape(B, 1, self.N, self.num_heads, self.dim_head).permute(0, 1, 3, 2, 4)
        return x
        
class LocalContext_Interaction(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 ):
        """
        Args:
            dim: feature size dimension.
            num_heads: number of attention head.
            window_size: window size.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            attn_drop: attention dropout rate.
            proj_drop: output dropout rate.
        """

        super().__init__()
        self.dim = dim
        window_size = (window_size,window_size)
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x,q_global):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class GlobalContext_Interaction(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 ):
        """
        Args:
            dim: feature size dimension.
            num_heads: number of attention head.
            window_size: window size.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            attn_drop: attention dropout rate.
            proj_drop: output dropout rate.
        """

        super().__init__()
        window_size = (window_size, window_size)
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        self.qkv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, q_global):
        B_, N, C = x.shape
        B = q_global.shape[0]
        kv = self.qkv(x).reshape(B_, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        q_global = q_global.repeat(1, B_ // B, 1, 1, 1)
        q = q_global.reshape(B_, self.num_heads, N, C // self.num_heads)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Context_Interaction_Block(nn.Module):

    def __init__(self,
                 latent_dim,
                 dim,
                 num_heads,
                 window_size=8,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 attention=LocalContext_Interaction,
                 norm_layer=nn.LayerNorm,
                 ):


        super().__init__()
        self.window_size = window_size
        self.norm1 = norm_layer(dim)

        self.attn = attention(
                              dim,
                              num_heads=num_heads,
                              window_size=window_size,
                              qkv_bias=qkv_bias,
                              qk_scale=qk_scale,
                              attn_drop=attn_drop,
                              proj_drop=drop,
                              )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.layer_scale = False

        self.gamma1 = 1.0
        self.gamma2 = 1.0
    
    def forward(self, x,q_global):
            B,H, W,C = x.shape
            shortcut = x
            x = self.norm1(x)
            x_windows = window_partition(x, self.window_size)
            x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
            attn_windows = self.attn(x_windows,q_global)
            x = window_reverse(attn_windows, self.window_size, H, W)
            x = shortcut + self.drop_path(self.gamma1 * x)
            x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
            return x

class Context_Interaction_layer(nn.Module):
    def __init__(self,n,latent_dim,in_channel,head,window_size,globalatten=False):
        super(Context_Interaction_layer,self).__init__()

        #layers=[]
        self.globalatten = globalatten
        self.model = nn.ModuleList([
            Context_Interaction_Block(
            latent_dim,
            in_channel,
            num_heads=head,
            window_size=window_size,
            attention=GlobalContext_Interaction if i%2 == 1 and self.globalatten == True else LocalContext_Interaction,
            )
            for i in range(n)])

        if self.globalatten == True:
            self.gen = Scale_aware_Query(latent_dim,in_channel,window_size=8,num_heads=head)
    def forward(self,x,latent):
        if self.globalatten == True:
            q_global = self.gen(latent)
            x = _to_channel_last(x)
            for model in self.model:
                x = model(x, q_global)
        else:
            x = _to_channel_last(x)
            for model in self.model:
                x = model(x,1)
        return x
def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)


##########################################################################
## Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=4, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel = channel
        self.reduction = reduction
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class SALayer(nn.Module):
    def __init__(self, channel,reduction=16):
        super(SALayer, self).__init__()
        self.pa = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, 1, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
    def forward(self, x):
        y = self.pa(x)
        return x * y

class Refine_Block(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(Refine_Block, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res

class Refine(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, num_cab):
        super(Refine, self).__init__()
        modules_body = []
        modules_body = [Refine_Block(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(num_cab)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

class HFPH(nn.Module):
    def __init__(self, n_feat,fusion_dim,  kernel_size, reduction, act, bias, num_cab):
        super(HFPH, self).__init__()
        self.refine0 = Refine(fusion_dim, kernel_size, reduction, act, bias, num_cab)
        self.refine1 = Refine(fusion_dim, kernel_size, reduction, act, bias, num_cab)
        self.refine2 = Refine(fusion_dim, kernel_size, reduction, act, bias, num_cab)
        self.refine3 = Refine(fusion_dim, kernel_size, reduction, act, bias, num_cab)

        self.up_enc1 = UpSample(n_feat[1],fusion_dim,s_factor=2)
        self.up_dec1 = UpSample(n_feat[1],fusion_dim,s_factor=2)

        self.up_enc2 = UpSample(n_feat[2],fusion_dim,s_factor=4)
        self.up_dec2 = UpSample(n_feat[2],fusion_dim,s_factor=4)

        self.up_enc3 = UpSample(n_feat[3],fusion_dim,s_factor=8)
        self.up_dec3 = UpSample(n_feat[3],fusion_dim,s_factor=8)

        layer0=[]
        for i in range(2):
            layer0.append(CALayer(fusion_dim,16))
            layer0.append(SALayer(fusion_dim,16))
        self.conv_enc0 = nn.Sequential(*layer0)

        layer1=[]
        for i in range(2):
            layer1.append(CALayer(fusion_dim,16))
            layer1.append(SALayer(fusion_dim,16))
        self.conv_enc1 = nn.Sequential(*layer1)

        layer2=[]
        for i in range(2):
            layer2.append(CALayer(fusion_dim,16))
            layer2.append(SALayer(fusion_dim,16))
        self.conv_enc2 = nn.Sequential(*layer2)

        layer3=[]
        for i in range(2):
            layer3.append(CALayer(fusion_dim,16))
            layer3.append(SALayer(fusion_dim,16))
        self.conv_enc3 = nn.Sequential(*layer3)

    def forward(self, x, encoder_outs, decoder_outs):
        x = x + self.conv_enc0(encoder_outs[0] + decoder_outs[3])
        x = self.refine0(x)
       
        x = x + self.conv_enc1(self.up_enc1(encoder_outs[1]) + self.up_dec1(decoder_outs[2]))
        x = self.refine1(x)
        
        x = x + self.conv_enc2(self.up_enc2(encoder_outs[2]) + self.up_dec2(decoder_outs[1]))
        x = self.refine2(x)
        
        x = x + self.conv_enc3(self.up_enc3(encoder_outs[3]) + self.up_dec3(decoder_outs[0]))
        x = self.refine3(x)
        
        return x

class Transformer_block(nn.Module):
    def __init__(self,dim,num_head=8,groups=2,qkv_bias=False,qk_scale=None,attn_drop=0.,proj_drop=0.,FFN_expand=2,norm_layer='WithBias',act_layer=nn.GELU,l_drop=0.,mlp_ratio=2,drop_path=0.,sr=1):
        super(Transformer_block,self).__init__()
        self.dim=dim*2
        self.num_head = num_head

        self.norm1 = LayerNorm(self.dim//2, norm_layer)
        self.norm3 = LayerNorm(self.dim//2, norm_layer)

        self.attn_nn = Attention(dim=self.dim//2,num_head=num_head,qkv_bias=qkv_bias,qk_scale=qk_scale,attn_drop=attn_drop,proj_drop=proj_drop,sr_ratio=sr)

        self.ffn = MFFN(self.dim//2, FFN_expand=2,norm_layer=nn.LayerNorm)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.apply(self._init_weights)
    
    def _init_weights(self,m):
        if isinstance(m,nn.Linear):
            trunc_normal_(m.weight,std=0.02)
            if isinstance(m,nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias,0)
        elif isinstance(m,nn.LayerNorm):
            nn.init.constant_(m.bias,0)
            nn.init.constant_(m.weight,1.0)
        elif isinstance(m,nn.Conv2d):
            fan_out = m.kernel_size[0]*m.kernel_size[1]*m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0,math.sqrt(2.0/fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        
    def forward(self,x):
        ind = x
        b,c,h,w = x.shape
        b,c,h,w = x.shape
        x =  self.attn_nn(self.norm1(x).reshape(b,c,h*w).transpose(1,2),h,w)
        b,c,h,w = x.shape
        x = self.drop_path(x)
        x = x+self.drop_path(self.ffn(self.norm3(x)))
        return x
        
class Transformer(nn.Module):
    def __init__(self,
    in_channels=3,
    out_cahnnels=3,
    transformer_blocks = 8,
    dim=[16,32,64,128,256],
    window_size = [8,8,8,8],
    patch_size = 64,
    swin_num = [4,6,7,8],
    head = [1,2,4,8,16],
    FFN_expand_=2,
    qkv_bias_=False,
    qk_sacle_=None,
    attn_drop_=0.,
    proj_drop_=0.,
    norm_layer_= 'WithBias',
    act_layer_=nn.GELU,
    l_drop_=0.,
    drop_path_=0.,
    sr=1,
    conv_num = [4,6,7,8],
    expand_ratio = [1,2,2,2],
    VAN = False,
    dynamic_e = False,
    dynamic_d = False,
    global_atten = True,
    ):
        super(Transformer,self).__init__()
        self.patch_size = patch_size
        
        self.embed = Down(in_channels,dim[0],3,1,1)
        self.conv0 = nn.Conv2d(dim[0],dim[4],1)
        self.encoder_level0 = nn.Sequential(Conv_block(conv_num[0],dim[0],dim[0],expand_ratio=expand_ratio[0],VAN=VAN,dynamic=dynamic_e))

        self.down0 = Down(dim[0],dim[1],3,2,1) ## H//2,W//2
        self.conv1 = nn.Conv2d(dim[1],dim[4],1)
        self.encoder_level1 = nn.Sequential(Conv_block(conv_num[1],dim[1],dim[1],expand_ratio=expand_ratio[1],VAN=VAN,dynamic=dynamic_e))
        
        self.down1 = Down(dim[1],dim[2],3,2,1)  ## H//4,W//4
        self.conv2 = nn.Conv2d(dim[2],dim[4],1)
        self.encoder_level2 = nn.Sequential(Conv_block(conv_num[2],dim[2],dim[2],expand_ratio=expand_ratio[2],VAN=VAN,dynamic=dynamic_e))
        
        self.down2 = Down(dim[2],dim[3],3,2,1)  ## H//8,W//8
        self.conv3 = nn.Conv2d(dim[3],dim[4],1)
        self.encoder_level3 = nn.Sequential(Conv_block(conv_num[3],dim[3],dim[3],expand_ratio=expand_ratio[3],VAN=VAN,dynamic=dynamic_e))
        
        self.down3 = Down(dim[3],dim[4],3,2,1) ## H//16,W//16

        self.latent = nn.Sequential(*[Transformer_block(dim=(dim[4]),num_head=head[4],qkv_bias=qkv_bias_,qk_scale=qk_sacle_,attn_drop=attn_drop_,proj_drop=proj_drop_,FFN_expand=FFN_expand_,norm_layer=norm_layer_,act_layer=act_layer_,l_drop=l_drop_,drop_path=drop_path_,sr=sr) for i in range(transformer_blocks)])
        
        self.up3 = Up((dim[4]),dim[3],4,2,1)
        self.ca3 = CALayer(dim[3]*2,reduction=4)
        self.reduce_chan_level3 = nn.Conv2d(dim[3]*2, dim[3], kernel_size=1, bias=False)
        self.decoder_level3 = Context_Interaction_layer(n=swin_num[3],latent_dim=dim[4],in_channel=dim[3],head=head[3],window_size=window_size[3],globalatten=global_atten)
        self.up2 = Up(dim[3],dim[2],4,2,1)
        self.ca2 = CALayer(dim[2]*2,reduction=4)
        self.reduce_chan_level2 = nn.Conv2d(dim[2]*2, dim[2], kernel_size=1, bias=False)
        self.decoder_level2 = Context_Interaction_layer(n=swin_num[2],latent_dim=dim[4],in_channel=dim[2],head=head[2],window_size=window_size[2],globalatten=global_atten)

        self.up1 = Up(dim[2],dim[1],4,2,1)
        self.ca1 = CALayer(dim[1]*2,reduction=4)
        self.reduce_chan_level1 = nn.Conv2d(dim[1]*2, dim[1], kernel_size=1, bias=False)
        self.decoder_level1 = Context_Interaction_layer(n=swin_num[1],latent_dim=dim[4],in_channel=dim[1],head=head[1],window_size=window_size[1],globalatten=global_atten)

        self.up0 = Up(dim[1],dim[0],4,2,1)
        self.ca0 = CALayer(dim[0]*2,reduction=4)
        self.reduce_chan_level0 = nn.Conv2d(dim[0]*2, dim[0], kernel_size=1, bias=False)
        self.decoder_level0 = Context_Interaction_layer(n=swin_num[0],latent_dim=dim[4],in_channel=dim[0],head=head[0],window_size=window_size[0],globalatten=global_atten)
        
        self.refinement = HFPH(n_feat=dim,fusion_dim=dim[0],kernel_size=3,reduction=4,act=nn.GELU(),bias=True,num_cab=6)

        self.out2 = nn.Conv2d(dim[0],out_cahnnels,kernel_size=3,stride=1,padding=1)

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        mod_pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x
    def forward(self,x):
        x = self.check_image_size(x)
        encoder_item = []
        decoder_item = []
        inp_enc_level0 = self.embed(x)
        inp_enc_level0 = self.encoder_level0(inp_enc_level0)
        encoder_item.append(inp_enc_level0)

        inp_enc_level1 = self.down0(inp_enc_level0)
        inp_enc_level1 = self.encoder_level1(inp_enc_level1)
        encoder_item.append(inp_enc_level1)

        inp_enc_level2 = self.down1(inp_enc_level1)
        inp_enc_level2 = self.encoder_level2(inp_enc_level2)
        encoder_item.append(inp_enc_level2)
        
        inp_enc_level3 = self.down2(inp_enc_level2)
        inp_enc_level3 = self.encoder_level3(inp_enc_level3)
        encoder_item.append(inp_enc_level3)

        out_enc_level4 = self.down3(inp_enc_level3)
        top_0 = F.adaptive_max_pool2d(inp_enc_level0,(out_enc_level4.shape[2],out_enc_level4.shape[3]))
        top_1 = F.adaptive_max_pool2d(inp_enc_level1,(out_enc_level4.shape[2],out_enc_level4.shape[3]))
        top_2 = F.adaptive_max_pool2d(inp_enc_level2,(out_enc_level4.shape[2],out_enc_level4.shape[3]))
        top_3 = F.adaptive_max_pool2d(inp_enc_level3,(out_enc_level4.shape[2],out_enc_level4.shape[3]))

        
        latent = out_enc_level4+self.conv0(top_0)+self.conv1(top_1)+self.conv2(top_2)+self.conv3(top_3)
        
        latent = self.latent(latent)

        inp_dec_level3 = self.up3(latent)
        inp_dec_level3 = F.upsample(inp_dec_level3,(inp_enc_level3.shape[2],inp_enc_level3.shape[3]),mode="bicubic")
        inp_dec_level3 = torch.cat([inp_dec_level3, inp_enc_level3], 1)
        inp_dec_level3 = self.ca3(inp_dec_level3)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)

        out_dec_level3 = self.decoder_level3(inp_dec_level3,latent) 
        out_dec_level3 = _to_channel_first(out_dec_level3)
        decoder_item.append(out_dec_level3)

        inp_dec_level2 = self.up2(out_dec_level3)
        inp_dec_level2 = F.upsample(inp_dec_level2,(inp_enc_level2.shape[2],inp_enc_level2.shape[3]),mode="bicubic")
        inp_dec_level2 = torch.cat([inp_dec_level2, inp_enc_level2], 1)
        inp_dec_level2 = self.ca2(inp_dec_level2)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2,latent)
        out_dec_level2 = _to_channel_first(out_dec_level2)
        decoder_item.append(out_dec_level2)

        inp_dec_level1 = self.up1(out_dec_level2)
        inp_dec_level1 = F.upsample(inp_dec_level1,(inp_enc_level1.shape[2],inp_enc_level1.shape[3]),mode="bicubic")
        inp_dec_level1 = torch.cat([inp_dec_level1, inp_enc_level1], 1)
        inp_dec_level1 = self.ca1(inp_dec_level1)
        inp_dec_level1 = self.reduce_chan_level1(inp_dec_level1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1,latent)
        out_dec_level1 = _to_channel_first(out_dec_level1)
        decoder_item.append(out_dec_level1)

        inp_dec_level0 = self.up0(out_dec_level1)
        inp_dec_level0 = F.upsample(inp_dec_level0,(inp_enc_level0.shape[2],inp_enc_level0.shape[3]),mode="bicubic")
        inp_dec_level0 = torch.cat([inp_dec_level0, inp_enc_level0], 1)
        inp_dec_level0 = self.ca0(inp_dec_level0)
        inp_dec_level0 = self.reduce_chan_level0(inp_dec_level0)
        out_dec_level0 = self.decoder_level0(inp_dec_level0,latent) 
        out_dec_level0 = _to_channel_first(out_dec_level0)
        decoder_item.append(out_dec_level0)

        out_dec_level0_refine = self.refinement(out_dec_level0,encoder_item,decoder_item)
        out_dec_level1 = self.out2(out_dec_level0_refine) + x
        
        return out_dec_level1
# from ptflops import get_model_complexity_info

# model = Transformer().cuda()
# H,W=256,256
# flops_t, params_t = get_model_complexity_info(model, (3, H,W), as_strings=True, print_per_layer_stat=True)

# print(f"net flops:{flops_t} parameters:{params_t}")
# model = nn.DataParallel(model)
# x = torch.ones([1,3,H,W]).cuda()
# b = model(x)
# steps=25
# # print(b)
# time_avgs=[]
# with torch.no_grad():
#     for step in range(steps):
        
#         torch.cuda.synchronize()
#         start = time.time()
#         result = model(x)
#         torch.cuda.synchronize()
#         time_interval = time.time() - start
#         if step>5:
#             time_avgs.append(time_interval)
#         #print('run time:',time_interval)
# print('avg time:',np.mean(time_avgs),'fps:',(1/np.mean(time_avgs)),' size:',H,W)
