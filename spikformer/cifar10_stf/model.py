import torch, math
import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepLIFNode, MultiStepParametricLIFNode
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import torch.nn.functional as F
from functools import partial

from torch import Tensor
from einops import rearrange

__all__ = ['spikformer']

# changed on 2025-08-15, 计算QKAttention中的动态连接
class DynamiceResidualBlock(nn.Module):
    def __init__(self, lidx: int, dim: int, C: int, last_layer: bool=False, expand_last: bool=False, round64: bool=True,) -> None:
        super().__init__()
        self.C = C if not last_layer else 1      # qkvr
        self.lidx = lidx
        self.dim = dim
        
        l = lidx + 2
        hid_dim, out_dim = l * self.C, l * self.C
        if last_layer and expand_last: hid_dim *= 4  
        if round64: hid_dim = (hid_dim// 64 +1) * 64 
        self.w1 = nn.Conv1d(self.dim, hid_dim, kernel_size=1, stride=1, bias=False)
        self.w1_bn = nn.BatchNorm1d(hid_dim)
        self.act_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.w2 = nn.Conv1d(hid_dim, out_dim, kernel_size=1, stride=1, bias=False)
        self.w2_bn = nn.BatchNorm1d(out_dim)
    
    def forward(self, x: Tensor) -> Tensor:

        T, B, C, H, W = x.shape
        x = x.flatten(3) # T B C N
        _, _, _, N = x.shape
        x_for_qkvr = x.flatten(0, 1) # (T B) C N

        x_for_qkvr = self.w1_bn(self.w1(x_for_qkvr)).reshape(T, B, -1, N)
        x_for_qkvr = self.act_lif(x_for_qkvr).flatten(0, 1) # (T B) C N
        x_for_qkvr = self.w2_bn(self.w2(x_for_qkvr)).reshape(T, B, -1, H, W) # T B (lidx+2)4 H W

        dw = rearrange(x_for_qkvr, 'T B (C L) H W -> C T B L H W', C=self.C)
        return dw   # 4 T B lidx+2 H W
    
    def layer_mix(self, hids, dw)-> Tensor:
        # dw [4 T B lidx+2 H W]  hids [T B C H W]
        x = tuple([sum(dw[cidx,:,:,j,None,:,:] * hids[j] for j in range(self.lidx+2)) for cidx in range(self.C)])
        return x    # [T B C H W] 共四组

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1_linear = nn.Linear(in_features, hidden_features)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc1_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.fc2_linear = nn.Linear(hidden_features, out_features)
        self.fc2_bn = nn.BatchNorm1d(out_features)
        self.fc2_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        
        # T,B,N,C = x.shape
        T, B, C, H, W = x.shape
        N = H*W
        x = x.flatten(-2).transpose(-1, -2)  # T,B,N,C
        x_ = x.flatten(0, 1)
        x = self.fc1_linear(x_)
        x = self.fc1_bn(x.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, self.c_hidden).contiguous()
        x = self.fc1_lif(x)

        x = self.fc2_linear(x.flatten(0,1))
        x = self.fc2_bn(x.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, C, H, W).contiguous()
        x = self.fc2_lif(x)
        return x


class SSA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.125
        self.q_linear = nn.Linear(dim, dim)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.k_linear = nn.Linear(dim, dim)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.v_linear = nn.Linear(dim, dim)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.attn_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend='cupy')

        self.proj_linear = nn.Linear(dim, dim)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            # T,B,N,C = x.shape
            T, B, C, H, W = x.shape
            N = H*W
            x = x.flatten(-2).transpose(-1, -2)     # T B N C

            x_for_qkv = x.flatten(0, 1)  # TB, N, C
            q_linear_out = self.q_linear(x_for_qkv)  # [TB, N, C]
            q_linear_out = self.q_bn(q_linear_out. transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C).contiguous()
            q_linear_out = self.q_lif(q_linear_out)
            q = q_linear_out.reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

            k_linear_out = self.k_linear(x_for_qkv)
            k_linear_out = self.k_bn(k_linear_out. transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C).contiguous()
            k_linear_out = self.k_lif(k_linear_out)
            k = k_linear_out.reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

            v_linear_out = self.v_linear(x_for_qkv)
            v_linear_out = self.v_bn(v_linear_out. transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C).contiguous()
            v_linear_out = self.v_lif(v_linear_out)
            v = v_linear_out.reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

            attn = (q @ k.transpose(-2, -1)) * self.scale
            x = attn @ v
            x = x.transpose(2, 3).reshape(T, B, N, C).contiguous()
            x = self.attn_lif(x)
            x = x.flatten(0, 1)
            x = self.proj_lif(self.proj_bn(self.proj_linear(x).transpose(-1, -2)).reshape(T, B, C, H, W))
            return x   # T B C H W
        elif isinstance(x, tuple):
            xq, xk, xv = x[0], x[1], x[2]
            T, B, C, H, W = xq.shape  # qkv shape is the same
            N = H*W
            xq = xq.flatten(3).transpose(-1, -2).flatten(0, 1)  # TB N C
            xk = xk.flatten(3).transpose(-1, -2).flatten(0, 1)
            xv = xv.flatten(3).transpose(-1, -2).flatten(0, 1)

            q_linear_out = self.q_linear(xq)  # [TB, N, C]
            q_linear_out = self.q_bn(q_linear_out. transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C).contiguous()
            q_linear_out = self.q_lif(q_linear_out)
            q = q_linear_out.reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

            k_linear_out = self.k_linear(xk)
            k_linear_out = self.k_bn(k_linear_out. transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C).contiguous()
            k_linear_out = self.k_lif(k_linear_out)
            k = k_linear_out.reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

            v_linear_out = self.v_linear(xv)
            v_linear_out = self.v_bn(v_linear_out. transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C).contiguous()
            v_linear_out = self.v_lif(v_linear_out)
            v = v_linear_out.reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

            attn = (q @ k.transpose(-2, -1)) * self.scale
            x = attn @ v
            x = x.transpose(2, 3).reshape(T, B, N, C).contiguous()
            x = self.attn_lif(x)
            x = x.flatten(0, 1)
            x = self.proj_lif(self.proj_bn(self.proj_linear(x).transpose(-1, -2)).reshape(T, B, C, H, W))
            return x   # T B C H W

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SSA(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):  
        if isinstance(x, torch.Tensor):
            ## T B C H W
            x = x + self.attn(x)
            x = x + self.mlp(x)
            return x    # T B C H W
        elif isinstance(x, tuple):
            # T B C H W
            res = x[-1]
            h = res + self.attn(x)
            out = h + self.mlp(h)
            return out  # T B C H W


class SPS(nn.Module):
    def __init__(self, img_size_h=128, img_size_w=128, patch_size=4, in_channels=2, embed_dims=256, 
                 recurrent_coding=False, recurrent_lif=None, time_step=None, pe_type=None,
                 temporal_conv_type=None):          # changed on 2025-04-13
        super().__init__()
        # changed on 2025-05-01
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32  # 默认为 float32，后续可能会使用混合精度

        self.image_size = [img_size_h, img_size_w]
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.C = in_channels
        self.embed_dims = embed_dims
        self.H, self.W = self.image_size[0] // patch_size[0], self.image_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj_conv = nn.Conv2d(in_channels, embed_dims//8, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dims//8)
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.proj_conv1 = nn.Conv2d(embed_dims//8, embed_dims//4, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn1 = nn.BatchNorm2d(embed_dims//4)
        self.proj_lif1 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.proj_conv2 = nn.Conv2d(embed_dims//4, embed_dims//2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn2 = nn.BatchNorm2d(embed_dims//2)
        self.proj_lif2 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj_conv3 = nn.Conv2d(embed_dims//2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn3 = nn.BatchNorm2d(embed_dims)
        self.proj_lif3 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.maxpool3 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.rpe_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.rpe_bn = nn.BatchNorm2d(embed_dims)
        self.rpe_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        # changed on 2025-04-13
        self.recurrent_coding = recurrent_coding
        self.recurrent_lif = recurrent_lif
        self.pe_type = pe_type
        self.temporal_conv_type = temporal_conv_type


        if recurrent_coding:
            if self.pe_type == "stf_1":
                if self.temporal_conv_type == "conv1d":
                    self.proj_temporal_conv = nn.Conv1d(embed_dims // 8, in_channels, kernel_size=1, stride=1, bias=False)
                    self.proj_temporal_bn = nn.BatchNorm1d(in_channels)
                elif self.temporal_conv_type == "conv2d":
                    self.proj_temporal_conv = nn.Conv2d(embed_dims // 8, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
                    self.proj_temporal_bn = nn.BatchNorm2d(in_channels)

            elif self.pe_type == "stf_2":
                if self.temporal_conv_type == "conv1d":
                    self.proj_temporal_conv = nn.Conv1d(embed_dims // 8, embed_dims // 8, kernel_size=1, stride=1, bias=False)
                    self.proj_temporal_bn = nn.BatchNorm1d(embed_dims // 8)
                elif self.temporal_conv_type == "conv2d":
                    self.proj_temporal_conv = nn.Conv2d(embed_dims // 8, embed_dims // 8, kernel_size=3, stride=1, padding=1, bias=False)
                    self.proj_temporal_bn = nn.BatchNorm2d(embed_dims // 8)

            if recurrent_lif is not None:
                if recurrent_lif == 'lif':
                    self.proj_temporal_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
                elif recurrent_lif == 'plif':
                    self.proj_temporal_lif = MultiStepParametricLIFNode(init_tau=2.0,detach_reset=True, backend='cupy')
        

    def forward(self, x):
        T, B, C, H, W = x.shape

        if not self.recurrent_coding:
            x = self.proj_conv(x.flatten(0, 1)) # have some fire value
            x = self.proj_bn(x).reshape(T, B, -1, H, W).contiguous()
            x = self.proj_lif(x).flatten(0, 1).contiguous()

        elif self.recurrent_coding:
            t_x = []
            
            if self.pe_type == "stf_1":
                for i in range(T):
                    if i == 0:
                        x_in = x[i] # B C H W
                    else:
                        x_in = x[i] + x_out
                    
                    x_out = self.proj_conv(x_in)    # B C H W
                    x_out = self.proj_bn(x_out)     # B C H W

                    x_out = self.proj_lif(x_out.unsqueeze(0)).squeeze(0)  # B C H W
                
                    tmp = x_out
                    
                    if self.temporal_conv_type == "conv1d":
                        x_out = x_out.flatten(2) # B C N
                        x_out = self.proj_temporal_conv(x_out)  # B C N
                        x_out = self.proj_temporal_bn(x_out).reshape(B, C, H, W).contiguous()

                    elif self.temporal_conv_type == "conv2d":
                        x_out = self.proj_temporal_conv(x_out)    # B C H W
                        x_out = self.proj_temporal_bn(x_out)     # B C H W

                    if self.recurrent_lif is not None:
                        x_out = self.proj_temporal_lif(x_out.unsqueeze(0)).squeeze(0) # B C H W

                    t_x.append(tmp)

            elif self.pe_type == "stf_2":
                x = self.proj_conv(x.flatten(0, 1)) # TB C H W
                x = self.proj_bn(x).reshape(T, B, -1, H, W) # T B C H W

                for i in range(T):
                    if i == 0:
                        x_in = x[i] # B C H W
                    else:
                        x_in = x[i] + x_out
                        # x_in = x_out
                
                    x_out = self.proj_lif(x_in.unsqueeze(0)).squeeze(0)  # B C H W
                
                    tmp = x_out
                    
                    if self.temporal_conv_type == "conv1d":
                        _, feat, _, _ = x_out.shape
                        x_out = x_out.flatten(2) # B C N
                        x_out = self.proj_temporal_conv(x_out)  # B C N
                        x_out = self.proj_temporal_bn(x_out).reshape(B, feat, H, W).contiguous()

                    elif self.temporal_conv_type == "conv2d":
                        x_out = self.proj_temporal_conv(x_out)    # B C H W
                        x_out = self.proj_temporal_bn(x_out)     # B C H W

                    if self.recurrent_lif is not None:
                        x_out = self.proj_temporal_lif(x_out.unsqueeze(0)).squeeze(0) # B C H W
                
                    t_x.append(tmp)
            x = torch.stack(t_x, dim=0).flatten(0, 1) # TB C H W
            

        x = self.proj_conv1(x)
        x = self.proj_bn1(x).reshape(T, B, -1, H, W).contiguous()
        x = self.proj_lif1(x).flatten(0, 1).contiguous()

        x = self.proj_conv2(x)
        x = self.proj_bn2(x).reshape(T, B, -1, H, W).contiguous()
        x = self.proj_lif2(x).flatten(0, 1).contiguous()
        x = self.maxpool2(x)

        x = self.proj_conv3(x)
        x = self.proj_bn3(x).reshape(T, B, -1, H//2, W//2).contiguous()
        x = self.proj_lif3(x).flatten(0, 1).contiguous()
        x = self.maxpool3(x)

        x_feat = x.reshape(T, B, -1, H//4, W//4).contiguous()
        x = self.rpe_conv(x)
        x = self.rpe_bn(x).reshape(T, B, -1, H//4, W//4).contiguous()
        x = self.rpe_lif(x)
        x = x + x_feat

        # x = x.flatten(-2).transpose(-1, -2)  # T,B,N,C
        # T B C H W
        return x






class SPS_Maxpooling_LIF_changed(nn.Module):
    def __init__(self, img_size_h=128, img_size_w=128, patch_size=4, in_channels=2, embed_dims=256, 
                 recurrent_coding=False, recurrent_lif=None, time_step=None, pe_type=None,
                 temporal_conv_type=None):          # changed on 2025-04-13
        super().__init__()
        # changed on 2025-05-01
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32  # 默认为 float32，后续可能会使用混合精度

        self.image_size = [img_size_h, img_size_w]
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.C = in_channels
        self.embed_dims = embed_dims
        self.H, self.W = self.image_size[0] // patch_size[0], self.image_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj_conv = nn.Conv2d(in_channels, embed_dims//8, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dims//8)
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.proj_conv1 = nn.Conv2d(embed_dims//8, embed_dims//4, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn1 = nn.BatchNorm2d(embed_dims//4)
        self.proj_lif1 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.proj_conv2 = nn.Conv2d(embed_dims//4, embed_dims//2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn2 = nn.BatchNorm2d(embed_dims//2)
        self.proj_lif2 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj_conv3 = nn.Conv2d(embed_dims//2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn3 = nn.BatchNorm2d(embed_dims)
        self.proj_lif3 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.maxpool3 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.rpe_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.rpe_bn = nn.BatchNorm2d(embed_dims)
        self.rpe_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        # changed on 2025-04-13
        self.recurrent_coding = recurrent_coding
        self.recurrent_lif = recurrent_lif
        self.pe_type = pe_type
        self.temporal_conv_type = temporal_conv_type


        if recurrent_coding:
            if self.pe_type == "stf_1":
                if self.temporal_conv_type == "conv1d":
                    self.proj_temporal_conv = nn.Conv1d(embed_dims // 8, in_channels, kernel_size=1, stride=1, bias=False)
                    self.proj_temporal_bn = nn.BatchNorm1d(in_channels)
                elif self.temporal_conv_type == "conv2d":
                    self.proj_temporal_conv = nn.Conv2d(embed_dims // 8, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
                    self.proj_temporal_bn = nn.BatchNorm2d(in_channels)

            elif self.pe_type == "stf_2":
                if self.temporal_conv_type == "conv1d":
                    self.proj_temporal_conv = nn.Conv1d(embed_dims // 8, embed_dims // 8, kernel_size=1, stride=1, bias=False)
                    self.proj_temporal_bn = nn.BatchNorm1d(embed_dims // 8)
                elif self.temporal_conv_type == "conv2d":
                    self.proj_temporal_conv = nn.Conv2d(embed_dims // 8, embed_dims // 8, kernel_size=3, stride=1, padding=1, bias=False)
                    self.proj_temporal_bn = nn.BatchNorm2d(embed_dims // 8)

            if recurrent_lif is not None:
                if recurrent_lif == 'lif':
                    self.proj_temporal_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
                elif recurrent_lif == 'plif':
                    self.proj_temporal_lif = MultiStepParametricLIFNode(init_tau=2.0,detach_reset=True, backend='cupy')
        

    def forward(self, x):
        T, B, C, H, W = x.shape

        if not self.recurrent_coding:
            x = self.proj_conv(x.flatten(0, 1)) # have some fire value
            x = self.proj_bn(x).reshape(T, B, -1, H, W).contiguous()
            x = self.proj_lif(x).flatten(0, 1).contiguous()

        elif self.recurrent_coding:
            t_x = []
            
            if self.pe_type == "stf_1":
                for i in range(T):
                    if i == 0:
                        x_in = x[i] # B C H W
                    else:
                        x_in = x[i] + x_out
                    
                    x_out = self.proj_conv(x_in)    # B C H W
                    x_out = self.proj_bn(x_out)     # B C H W

                    x_out = self.proj_lif(x_out.unsqueeze(0)).squeeze(0)  # B C H W
                
                    tmp = x_out
                    
                    if self.temporal_conv_type == "conv1d":
                        x_out = x_out.flatten(2) # B C N
                        x_out = self.proj_temporal_conv(x_out)  # B C N
                        x_out = self.proj_temporal_bn(x_out).reshape(B, C, H, W).contiguous()

                    elif self.temporal_conv_type == "conv2d":
                        x_out = self.proj_temporal_conv(x_out)    # B C H W
                        x_out = self.proj_temporal_bn(x_out)     # B C H W

                    if self.recurrent_lif is not None:
                        x_out = self.proj_temporal_lif(x_out.unsqueeze(0)).squeeze(0) # B C H W

                    t_x.append(tmp)

            elif self.pe_type == "stf_2":
                x = self.proj_conv(x.flatten(0, 1)) # TB C H W
                x = self.proj_bn(x).reshape(T, B, -1, H, W) # T B C H W

                for i in range(T):
                    if i == 0:
                        x_in = x[i] # B C H W
                    else:
                        x_in = x[i] + x_out
                        # x_in = x_out
                
                    x_out = self.proj_lif(x_in.unsqueeze(0)).squeeze(0)  # B C H W
                
                    tmp = x_out
                    
                    if self.temporal_conv_type == "conv1d":
                        _, feat, _, _ = x_out.shape
                        x_out = x_out.flatten(2) # B C N
                        x_out = self.proj_temporal_conv(x_out)  # B C N
                        x_out = self.proj_temporal_bn(x_out).reshape(B, feat, H, W).contiguous()

                    elif self.temporal_conv_type == "conv2d":
                        x_out = self.proj_temporal_conv(x_out)    # B C H W
                        x_out = self.proj_temporal_bn(x_out)     # B C H W

                    if self.recurrent_lif is not None:
                        x_out = self.proj_temporal_lif(x_out.unsqueeze(0)).squeeze(0) # B C H W
                
                    t_x.append(tmp)
            x = torch.stack(t_x, dim=0).flatten(0, 1) # TB C H W
            

        x = self.proj_conv1(x)
        x = self.proj_bn1(x).reshape(T, B, -1, H, W).contiguous()
        x = self.proj_lif1(x).flatten(0, 1).contiguous()

        x = self.proj_conv2(x)
        # maxpooling then lif
        x = self.proj_bn2(x).reshape(T, B, -1, H, W).flatten(0, 1).contiguous()
        x = self.maxpool2(x).reshape(T, B, -1, H//2, W//2).contiguous()
        x = self.proj_lif2(x).flatten(0, 1).contiguous()


        x = self.proj_conv3(x)
        # maxpooling then lif
        x = self.proj_bn3(x).reshape(T, B, -1, H//2, W//2).flatten(0, 1).contiguous()
        x = self.maxpool3(x).reshape(T, B, -1, H//4, W//4).contiguous()
        x = self.proj_lif3(x).flatten(0, 1).contiguous()

        x_feat = x.reshape(T, B, -1, H//4, W//4).contiguous()
        x = self.rpe_conv(x)
        x = self.rpe_bn(x).reshape(T, B, -1, H//4, W//4).contiguous()
        x = self.rpe_lif(x)
        x = x + x_feat

        # x = x.flatten(-2).transpose(-1, -2)  # T,B,N,C
        # T B C H W
        return x



class Spikformer(nn.Module):
    def __init__(self,
                 img_size_h=128, img_size_w=128, patch_size=16, in_channels=2, num_classes=11,
                 embed_dims=[64, 128, 256], num_heads=[1, 2, 4], mlp_ratios=[4, 4, 4], qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[6, 8, 6], sr_ratios=[8, 4, 2], T = 4, 
                 recurrent_coding=False, recurrent_lif=None, # changed on 2025-04-13
                 pe_type=None, temporal_conv_type=None, maxpooling_lif_change_order=False,
                 dense_connection=False,
                 ):
        super().__init__()
        self.T = T  # time step
        self.num_classes = num_classes
        self.depths = depths
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule
        
        assert pe_type in ("stf_1", "stf_2"), f"Invalid pe_type: {pe_type}, must be 'stf_1' or 'stf_2'"
        assert temporal_conv_type in ("conv1d", "conv2d"), f"Invalid temporal_conv_type: {pe_type}, must be 'conv1d' or 'conv2d'"

        self.recurrent_coding = recurrent_coding
        self.recurrent_lif = recurrent_lif
        self.pe_type = pe_type
        self.temporal_conv_type = temporal_conv_type
        self.maxpooling_lif_change_order = maxpooling_lif_change_order
        self.dense_connection = dense_connection

        if not maxpooling_lif_change_order:
            patch_embed = SPS(img_size_h=img_size_h,
                                    img_size_w=img_size_w,
                                    patch_size=patch_size,
                                    in_channels=in_channels,
                                    embed_dims=embed_dims,
                                    recurrent_coding=recurrent_coding,
                                    recurrent_lif=recurrent_lif,
                                    time_step=T,
                                    pe_type=pe_type,
                                    temporal_conv_type=temporal_conv_type)                # changed on 2025-04-13
        else:
            patch_embed = SPS_Maxpooling_LIF_changed(img_size_h=img_size_h,
                                    img_size_w=img_size_w,
                                    patch_size=patch_size,
                                    in_channels=in_channels,
                                    embed_dims=embed_dims,
                                    recurrent_coding=recurrent_coding,
                                    recurrent_lif=recurrent_lif,
                                    time_step=T,
                                    pe_type=pe_type,
                                    temporal_conv_type=temporal_conv_type)


        block = nn.ModuleList([Block(
            dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
            norm_layer=norm_layer, sr_ratio=sr_ratios)
            for j in range(depths)])

        if self.dense_connection:
            # Block qkvr C=4
            block_da = nn.ModuleList([DynamiceResidualBlock(lidx=lidx, dim=embed_dims, C=4, 
                last_layer=lidx==depths-1, expand_last=True, round64=True) for lidx in range(depths)])
            block_bs = nn.ParameterList([nn.Parameter(data=torch.randn(4 if lidx != depths-1 else 1, lidx+2)) for lidx in range(depths)])

        setattr(self, f"patch_embed", patch_embed)
        setattr(self, f"block", block)
        if self.dense_connection:
            setattr(self, f"block_da", block_da)
            setattr(self, f"block_bs", block_bs)

        # classification head
        self.head = nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    @torch.jit.ignore
    def _get_pos_embed(self, pos_embed, patch_embed, H, W):
        if H * W == self.patch_embed1.num_patches:
            return pos_embed
        else:
            return F.interpolate(
                pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(0, 3, 1, 2),
                size=(H, W), mode="bilinear").reshape(1, -1, H * W).permute(0, 2, 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x, use_dense_connection=False):

        block = getattr(self, f"block")
        patch_embed = getattr(self, f"patch_embed")
        if self.dense_connection and use_dense_connection:
            block_da = getattr(self, f"block_da")
            block_bs = getattr(self, f"block_bs")

        x = patch_embed(x)
        if self.dense_connection and use_dense_connection:
            hiddens = [x] # T B C H W
            idx = 0
        for blk in block:
            x = blk(x)
            if self.dense_connection and use_dense_connection:
                
                hiddens.append(x)   # T B C H W
                dw = block_da[idx](x) # 4 T B lidx+2 H W
                dw = dw + block_bs[idx][:, None, None, :, None, None]  # 4 T B lidx+2 H W
                x = block_da[idx].layer_mix(hiddens, dw)
                idx += 1
        if self.dense_connection and use_dense_connection:
            x = x[0]
            # T B C H W

        x = x.flatten(3).mean(3)    # T B C
        return x

    def forward(self, x, use_dense_connection=False):
        x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
        x = self.forward_features(x, use_dense_connection=use_dense_connection)
        x = self.head(x.mean(0))
        return x


@register_model
def spikformer(pretrained=False, **kwargs):
    model = Spikformer(
        # img_size_h=224, img_size_w=224,
        # patch_size=16, embed_dims=768, num_heads=12, mlp_ratios=4,
        # in_channels=3, num_classes=1000, qkv_bias=False,
        # norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=12, sr_ratios=1,
        **kwargs
    )
    model.default_cfg = _cfg()

    print(f"recurrent_coding: {model.recurrent_coding}")
    print(f"recurrent_lif: {model.recurrent_lif}")
    print(f"pe_type: {model.pe_type}")
    print(f"temporal_conv_type: {model.temporal_conv_type}")
    print(f"maxpooling_lif_change_orderL: {model.maxpooling_lif_change_order}")
    print(f"dense_connection: {model.dense_connection}")

    return model