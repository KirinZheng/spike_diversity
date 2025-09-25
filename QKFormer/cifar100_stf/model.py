import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepParametricLIFNode, MultiStepLIFNode
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from functools import partial
from timm.models import create_model
from torch import Tensor
from einops import rearrange

__all__ = ['QKFormer']

class InPlaceSetSlice(torch.autograd.Function):

    @staticmethod
    def forward(ctx, full_tensor, last_slice, x_idx, x_val):
        full_tensor[x_idx] = x_val
        ctx.x_idx = x_idx
        ret = torch.empty(0, device=full_tensor.device, dtype=full_tensor.dtype)
        ret.set_(full_tensor[:x_idx + 1])
        return ret

    @staticmethod
    def backward(ctx, grad_out):
        if ctx.x_idx == 0:
            return None, None, None, grad_out[ctx.x_idx]
        else:
            return None, grad_out[:ctx.x_idx], None, grad_out[ctx.x_idx]


## changed on 2025-09-16
def apply_inplace_set(x_acc, x_idx, x_val):
    full_tensor, last_slice = x_acc
    new_slice = InPlaceSetSlice.apply(full_tensor, last_slice, x_idx, x_val)
    return full_tensor, new_slice


# changed on 2025-08-15, 计算QKAttention中的动态连接
class DynamiceResidualBlock(nn.Module):
    def __init__(self, lidx: int, dim: int, last_layer: bool=False, expand_last: bool=False, round64: bool=True) -> None:
        super().__init__()
        self.C = 4 if not last_layer else 1      # qkvr
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


class Token_QK_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads

        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.attn_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend='cupy')

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')


    def forward(self, x):
        T, B, C, H, W = x.shape

        x = x.flatten(3)
        T, B, C, N = x.shape
        x_for_qkv = x.flatten(0, 1)

        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T, B, C, N)
        q_conv_out = self.q_lif(q_conv_out)
        q = q_conv_out.unsqueeze(2).reshape(T, B, self.num_heads, C // self.num_heads, N)

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T, B, C, N)
        k_conv_out = self.k_lif(k_conv_out)
        k = k_conv_out.unsqueeze(2).reshape(T, B, self.num_heads, C // self.num_heads, N)

        q = torch.sum(q, dim = 3, keepdim = True)
        attn = self.attn_lif(q)
        x = torch.mul(attn, k)

        x = x.flatten(2, 3)
        x = self.proj_bn(self.proj_conv(x.flatten(0, 1))).reshape(T, B, C, H, W)
        x = self.proj_lif(x)

        return x


class Spiking_Self_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = 0.125
        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.v_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.attn_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend='cupy')

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.qkv_mp = nn.MaxPool1d(4)

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            T, B, C, H, W = x.shape

            x = x.flatten(3)
            T, B, C, N = x.shape
            x_for_qkv = x.flatten(0, 1)

            q_conv_out = self.q_conv(x_for_qkv)
            q_conv_out = self.q_bn(q_conv_out).reshape(T,B,C,N).contiguous()
            q_conv_out = self.q_lif(q_conv_out)
            q = q_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

            k_conv_out = self.k_conv(x_for_qkv)
            k_conv_out = self.k_bn(k_conv_out).reshape(T,B,C,N).contiguous()
            k_conv_out = self.k_lif(k_conv_out)
            k = k_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

            v_conv_out = self.v_conv(x_for_qkv)
            v_conv_out = self.v_bn(v_conv_out).reshape(T,B,C,N).contiguous()
            v_conv_out = self.v_lif(v_conv_out)
            v = v_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

            x = k.transpose(-2,-1) @ v
            x = (q @ x) * self.scale

            x = x.transpose(3, 4).reshape(T, B, C, N).contiguous()
            x = self.attn_lif(x)
            x = x.flatten(0,1)
            x = self.proj_lif(self.proj_bn(self.proj_conv(x))).reshape(T,B,C,W,H)

            return x
        elif isinstance(x, tuple):
            xq, xk, xv = x[0], x[1], x[2]
            T, B, C, H, W = xq.shape        # qkv shape is the same
            N = H * W
            xq = xq.flatten(3).flatten(0, 1)
            xk = xk.flatten(3).flatten(0, 1)
            xv = xv.flatten(3).flatten(0, 1)

            q_conv_out = self.q_conv(xq)
            q_conv_out = self.q_bn(q_conv_out).reshape(T,B,C,N).contiguous()
            q_conv_out = self.q_lif(q_conv_out)
            q = q_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

            k_conv_out = self.k_conv(xk)
            k_conv_out = self.k_bn(k_conv_out).reshape(T,B,C,N).contiguous()
            k_conv_out = self.k_lif(k_conv_out)
            k = k_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

            v_conv_out = self.v_conv(xv)
            v_conv_out = self.v_bn(v_conv_out).reshape(T,B,C,N).contiguous()
            v_conv_out = self.v_lif(v_conv_out)
            v = v_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

            x = k.transpose(-2,-1) @ v
            x = (q @ x) * self.scale

            x = x.transpose(3, 4).reshape(T, B, C, N).contiguous()
            x = self.attn_lif(x)
            x = x.flatten(0,1)
            x = self.proj_lif(self.proj_bn(self.proj_conv(x))).reshape(T,B,C,W,H)

            return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.mlp1_conv = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1)
        self.mlp1_bn = nn.BatchNorm2d(hidden_features)
        self.mlp1_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.mlp2_conv = nn.Conv2d(hidden_features, out_features, kernel_size=1, stride=1)
        self.mlp2_bn = nn.BatchNorm2d(out_features)
        self.mlp2_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        T, B, C, H, W = x.shape

        x = self.mlp1_conv(x.flatten(0, 1))
        x = self.mlp1_bn(x).reshape(T, B, self.c_hidden, H, W)
        x = self.mlp1_lif(x)

        x = self.mlp2_conv(x.flatten(0, 1))
        x = self.mlp2_bn(x).reshape(T, B, C, H, W)
        x = self.mlp2_lif(x)

        return x


class TokenSpikingTransformer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.tssa = Token_QK_Attention(dim, num_heads)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features= dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):

        x = x + self.tssa(x)
        # print(torch.unique(x))
        x = x + self.mlp(x)
        # print(torch.unique(x))

        return x


class SpikingTransformer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.ssa = Spiking_Self_Attention(dim, num_heads)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features= dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            x = x + self.ssa(x)
            x = x + self.mlp(x)
            return x
        elif isinstance(x, tuple):
            res = x[-1]
            h = res + self.ssa(x)
            out = h + self.mlp(h)
            return out


class PatchEmbedInit(nn.Module):
    def __init__(self, img_size_h=128, img_size_w=128, patch_size=4, in_channels=2, embed_dims=256,
                 recurrent_coding=False, recurrent_lif=None, pe_type=None, time_step=None, temporal_conv_type=None):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.C = in_channels
        self.H, self.W = self.image_size[0] // patch_size[0], self.image_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj_conv = nn.Conv2d(in_channels, embed_dims // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dims // 2)
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.proj1_conv = nn.Conv2d(embed_dims // 2, embed_dims // 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj1_bn = nn.BatchNorm2d(embed_dims // 1)
        self.proj1_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.proj_res_conv = nn.Conv2d(embed_dims//2, embed_dims //1, kernel_size=1, stride=1, padding=0, bias=False)
        self.proj_res_bn = nn.BatchNorm2d(embed_dims)
        self.proj_res_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        ## changed on 2025-04-08
        self.recurrent_coding = recurrent_coding
        self.recurrent_lif = recurrent_lif
        self.pe_type = pe_type
        self.time_step = time_step
        self.temporal_conv_type = temporal_conv_type
        
        if recurrent_coding:
            if self.pe_type == "stf_1":
                if self.temporal_conv_type == "conv1d":
                    self.proj_temporal_conv = nn.Conv1d(embed_dims // 2, in_channels, kernel_size=1, stride=1, bias=False)
                    self.proj_temporal_bn = nn.BatchNorm1d(in_channels)
                elif self.temporal_conv_type == "conv2d":
                    self.proj_temporal_conv = nn.Conv2d(embed_dims // 2, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
                    self.proj_temporal_bn = nn.BatchNorm2d(in_channels)

            elif self.pe_type == "stf_2":
                if self.temporal_conv_type == "conv1d":
                    self.proj_temporal_conv = nn.Conv1d(embed_dims // 2, embed_dims // 2, kernel_size=1, stride=1, bias=False)
                    self.proj_temporal_bn = nn.BatchNorm1d(embed_dims // 2)
                elif self.temporal_conv_type == "conv2d":
                    self.proj_temporal_conv = nn.Conv2d(embed_dims // 2, embed_dims // 2, kernel_size=3, stride=1, padding=1, bias=False)
                    self.proj_temporal_bn = nn.BatchNorm2d(embed_dims // 2)

            if recurrent_lif is not None:
                if recurrent_lif == 'lif':
                    self.proj_temporal_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
                elif recurrent_lif == 'plif':
                    self.proj_temporal_lif = MultiStepParametricLIFNode(init_tau=2.0,detach_reset=True, backend='cupy')


    def forward(self, x):
        T, B, C, H, W = x.shape

        if self.recurrent_coding:
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

        elif not self.recurrent_coding:
            x = self.proj_conv(x.flatten(0, 1))
            x = self.proj_bn(x).reshape(T, B, -1, H, W)
            x = self.proj_lif(x).flatten(0, 1)

        x_feat = x
        x = self.proj1_conv(x)
        x = self.proj1_bn(x).reshape(T, B, -1, H, W)
        x = self.proj1_lif(x)

        x_feat = self.proj_res_conv(x_feat)
        x_feat = self.proj_res_bn(x_feat).reshape(T, B, -1, H, W).contiguous()
        x_feat = self.proj_res_lif(x_feat)

        x = x + x_feat # shortcut

        return x


class PatchEmbeddingStage(nn.Module):
    def __init__(self, img_size_h=128, img_size_w=128, patch_size=4, in_channels=2, embed_dims=256):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.C = in_channels
        self.H, self.W = self.image_size[0] // patch_size[0], self.image_size[1] // patch_size[1]
        self.num_patches = self.H * self.W

        self.proj3_conv = nn.Conv2d(embed_dims//2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj3_bn = nn.BatchNorm2d(embed_dims)
        self.proj3_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.proj4_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj4_bn = nn.BatchNorm2d(embed_dims)
        self.proj4_maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.proj4_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.proj_res_conv = nn.Conv2d(embed_dims//2, embed_dims, kernel_size=1, stride=2, padding=0, bias=False)
        self.proj_res_bn = nn.BatchNorm2d(embed_dims)
        self.proj_res_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

    def forward(self, x):
        T, B, C, H, W = x.shape
        # Downsampling + Res

        x = x.flatten(0, 1).contiguous()
        x_feat = x

        x = self.proj3_conv(x)
        x = self.proj3_bn(x).reshape(T, B, -1, H, W).contiguous()
        x = self.proj3_lif(x).flatten(0, 1).contiguous()

        x = self.proj4_conv(x)
        x = self.proj4_bn(x)
        x = self.proj4_maxpool(x).reshape(T, B, -1, H // 2, W // 2).contiguous()
        x = self.proj4_lif(x)

        x_feat = self.proj_res_conv(x_feat)
        x_feat = self.proj_res_bn(x_feat).reshape(T, B, -1, H//2, W//2).contiguous()
        x_feat = self.proj_res_lif(x_feat)

        x = x + x_feat # shortcut

        return x


class spiking_transformer(nn.Module):
    def __init__(self,
                 img_size_h=128, img_size_w=128, patch_size=16, in_channels=2, num_classes=11,
                 embed_dims=[64, 128, 256], num_heads=[8, 8, 8], mlp_ratios=[4, 4, 4], qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[6, 8, 6], sr_ratios=[8, 4, 2], T=4, pretrained_cfg=None,
                 recurrent_coding=False, recurrent_lif=None, pe_type=None,
                 temporal_conv_type=None, dense_connection=False,
                 dense_easy_connection=False,
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.T = T
        num_heads = [8, 8, 8]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule

        assert pe_type in ("stf_1", "stf_2"), f"Invalid pe_type: {pe_type}, must be 'stf_1' or 'stf_2'"
        assert temporal_conv_type in ("conv1d", "conv2d"), f"Invalid temporal_conv_type: {temporal_conv_type}, must be 'conv1d' or 'conv2d'"
        assert (not dense_connection and not dense_easy_connection) or (dense_connection ^ dense_easy_connection), \
                "Invalid config: set at most one of 'dense_connection' and 'dense_easy_connection' to True."

        self.recurrent_coding = recurrent_coding
        self.recurrent_lif = recurrent_lif
        self.pe_type = pe_type
        self.temporal_conv_type = temporal_conv_type
        self.dense_connection = dense_connection        # changed on 2025-09-22
        self.dense_easy_connection = dense_easy_connection

        patch_embed1 = PatchEmbedInit(img_size_h=img_size_h,
                                       img_size_w=img_size_w,
                                       patch_size=patch_size,
                                       in_channels=in_channels,
                                       embed_dims=embed_dims // 4,
                                       recurrent_coding=recurrent_coding,
                                       recurrent_lif=recurrent_lif,
                                       pe_type=pe_type,
                                       time_step=T,
                                       temporal_conv_type=temporal_conv_type)

        # stage1: QKFormers
        stage1 = nn.ModuleList([TokenSpikingTransformer(
            dim=embed_dims // 4, num_heads=num_heads[0], mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
            norm_layer=norm_layer, sr_ratio=sr_ratios)
            for j in range(1)])

        patch_embed2 = PatchEmbeddingStage(img_size_h=img_size_h,
                                       img_size_w=img_size_w,
                                       patch_size=patch_size,
                                       in_channels=in_channels,
                                       embed_dims=embed_dims // 2)

        stage2 = nn.ModuleList([TokenSpikingTransformer(
            dim=embed_dims // 2, num_heads=num_heads[1], mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
            norm_layer=norm_layer, sr_ratio=sr_ratios)
            for j in range(1)])

        patch_embed3 = PatchEmbeddingStage(img_size_h=img_size_h,
                                       img_size_w=img_size_w,
                                       patch_size=patch_size,
                                       in_channels=in_channels,
                                       embed_dims=embed_dims)

        stage3 = nn.ModuleList([SpikingTransformer(
            dim=embed_dims, num_heads=num_heads[2], mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
            norm_layer=norm_layer, sr_ratio=sr_ratios)
            for j in range(depths - 2)])

        # changed on 2025-09-22
        if self.dense_connection:
            stage3_da = nn.ModuleList([DynamiceResidualBlock(lidx=lidx, dim=embed_dims, 
                                                            last_layer=lidx==depths-2-1, expand_last=True, round64=True) for lidx in range(depths-2)])

            dense_bs = nn.ParameterList([nn.Parameter(data=torch.randn(4 if lidx != depths-2-1 else 1, lidx+2)) for lidx in range(depths-2)])

        if self.dense_easy_connection:
            self.n_repeat = (depths - 2) // 1
            self.dilation_factor = 1
            self.increate_T_every = 1
            self.weights = nn.ModuleList([
                    nn.Linear((i + 2 + self.dilation_factor - 1) // self.dilation_factor, 1, bias=False) 
                    for i in range(self.n_repeat)
                ])

        setattr(self, f"patch_embed1", patch_embed1)
        setattr(self, f"patch_embed2", patch_embed2)
        setattr(self, f"patch_embed3", patch_embed3)
        setattr(self, f"stage1", stage1)
        setattr(self, f"stage2", stage2)
        setattr(self, f"stage3", stage3)
        # changed on 2025-09-22
        if self.dense_connection:
            setattr(self, f"stage3_da", stage3_da)
            setattr(self, f"dense_bs", dense_bs)

        # classification head
        self.head = nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

        if self.dense_easy_connection:
            for module in self.weights:
                module.weight.data.zero_()
                module.weight.data[:,-1] = 1.

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x, use_dense_connection=False):
        stage1 = getattr(self, f"stage1")
        patch_embed1 = getattr(self, f"patch_embed1")
        stage2 = getattr(self, f"stage2")
        patch_embed2 = getattr(self, f"patch_embed2")
        stage3 = getattr(self, f"stage3")
        patch_embed3 = getattr(self, f"patch_embed3")

        # changed on 2025-09-22
        if self.dense_connection:
            stage3_da = getattr(self, f"stage3_da")
            dense_bs = getattr(self, f"dense_bs")

        x = patch_embed1(x)
        for blk in stage1:
            x = blk(x)

        x = patch_embed2(x)
        for blk in stage2:
            x = blk(x)

        x = patch_embed3(x)

        if self.dense_connection:
            # changed on 2025-09-22
            if use_dense_connection:
                hiddens = [x]  # [T B C H W]
                idx = 0

            for blk in stage3:
                x = blk(x)
                # changed on 2025-09-22
                if use_dense_connection:
                    hiddens.append(x)  # T B C H W
                    dw = stage3_da[idx](x)   # 4 T B lidx+2 H W
                    dw = dw + dense_bs[idx][:, None, None, :, None, None]  # 4 T B lidx+2 H W
                    x = stage3_da[idx].layer_mix(hiddens, dw)
                    idx += 1
            # changed on 2025-09-22
            if use_dense_connection:
                x = x[0]
                
        elif self.dense_easy_connection:
            if use_dense_connection:
                hiddens = []
                for i in range(self.dilation_factor):
                    current_group_size = (self.n_repeat + 1) // self.dilation_factor
                    if i < (self.n_repeat + 1) % self.dilation_factor:
                        current_group_size += 1
                    
                    hiddens.append((torch.zeros((current_group_size, *x.shape), device=x.device,
                        dtype=x.dtype),  None))
                
                # 加入patch_embed3中的输出
                hiddens[0] = apply_inplace_set(hiddens[0], 0, x)


                for rep_idx in range(1, self.n_repeat + 1):
                    for i in range(self.increate_T_every):
                        x = stage3[(rep_idx - 1) * self.increate_T_every  + i](x)
                    hiddens[rep_idx % self.dilation_factor] = apply_inplace_set(
                        hiddens[rep_idx % self.dilation_factor], 
                        rep_idx // self.dilation_factor, 
                        x,
                    )
                    x = torch.tensordot(self.weights[rep_idx - 1].weight.view(-1), 
                                        hiddens[rep_idx % self.dilation_factor][1], dims=1)
            else:
                for blk in stage3:
                    x = blk(x)
        # none any type of dynamic dense connection, baseline code
        else: 
            for blk in stage3:
                x = blk(x)

        return x.flatten(3).mean(3)

    def forward(self, x, use_dense_connection=False):
        x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
        x = self.forward_features(x, use_dense_connection=use_dense_connection)
        x = self.head(x.mean(0))

        return x


@register_model
def QKFormer(pretrained=False, **kwargs):
    model = spiking_transformer(
        **kwargs
    )
    model.default_cfg = _cfg()
    print(f"recurrent_coding: {model.recurrent_coding}")
    print(f"recurrent_lif: {model.recurrent_lif}")
    print(f"pe_type: {model.pe_type}")
    print(f"temporal_conv_type: {model.temporal_conv_type}")
    print(f"dense_connection: {model.dense_connection}")
    print(f"dense_easy_connection: {model.dense_easy_connection}")
    return model


if __name__ == '__main__':
    input = torch.randn(2, 3, 32, 32).cuda()
    model = create_model(
        'QKFormer',
        pretrained=False,
        drop_rate=0,
        drop_path_rate=0.1,
        drop_block_rate=None,
        img_size_h=32, img_size_w=32,
        patch_size=4, embed_dims=384, num_heads=8, mlp_ratios=4,
        in_channels=3, num_classes=100, qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=4, sr_ratios=1,
        T=4,
    ).cuda()

    from torchinfo import summary
    summary(model, input_size=(2, 3, 32, 32))