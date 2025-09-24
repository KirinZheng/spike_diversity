# from visualizer import get_local
import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepParametricLIFNode, MultiStepLIFNode
from spikingjelly.clock_driven import layer
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from functools import partial

from util.embeddings import get_3d_sincos_pos_embed, get_sinusoid_spatial_temporal_encoding
import math

from spikingjelly.clock_driven import surrogate, lava_exchange
from spikingjelly.clock_driven import neuron_kernel, cu_kernel_opt
from typing import Callable

__all__ = ['QKFormer_10_512',]


# changed on 2025-4-30
class MultiStepLazyStateLIFNode(MultiStepLIFNode):
    """
    >>>inputs = torch.randn((2, 1, 4))
    >>>node = LazyStateLIFNode()
    >>>out = node(inputs)
    >>>print(node.v)
    tensor([[[ 0.1676, -0.3076, -0.1530, -0.1675]],
            [[-0.0658, -1.4495, -0.3014, -0.2170]]])
    >>>node.reset()
    >>>print(node.v)
    tensor([[[0.4139, 0.1390, 0.8201, 0.3612]],
            [[0.3644, 0.9767, 0.0484, 0.7073]]], grad_fn=<AddBackward0>)
    """

    def __init__(self, tau: float = 2., decay_input: bool = True, v_threshold: float = 1.,
                 v_reset: float = 0., surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False, backend='torch', lava_s_cale=1 << 6,
                 sigmoid_init=False):
        """
        :param init_state_shape: (B, C, H, W) or (B, N, D)
                        x.shape: (T, B, C, H, W) or (T, B, N, D)
        """
        super(MultiStepLazyStateLIFNode, self).__init__(tau, decay_input, v_threshold, v_reset, surrogate_function,
                                               detach_reset, backend, lava_s_cale)
        self.init_state = None
        self.have_init = False
        self.init_func = torch.sigmoid if sigmoid_init else lambda x: x

    def check_init_state(self, x):
        if not self.have_init:
            self.init_state = nn.Parameter(nn.init.uniform_(
                torch.empty((1, *x.shape[1:]), device=x.device), a=0.4, b=0.6))
            self.have_init = True
        self.v = torch.broadcast_to(self.init_func(self.init_state), x.shape).to(x)

    def forward(self, input_tensor, temporal_fd_after_first=False):

        x_seq = input_tensor  # select first arguments

        # 如果初始化后没有init 并且 self.init_state为None
        if not temporal_fd_after_first:
            self.check_init_state(x_seq[0])  # 

        assert x_seq.dim() > 1
        # x_seq.shape = [T, *]

        if self.backend == 'torch':
            spike_seq = []
            self.v_seq = []
            for t in range(x_seq.shape[0]):
                spike_seq.append(super(MultiStepLIFNode, self).forward(x_seq[t]).unsqueeze(0))
                self.v_seq.append(self.v.unsqueeze(0))
            spike_seq = torch.cat(spike_seq, 0)
            self.v_seq = torch.cat(self.v_seq, 0)
            return spike_seq

        elif self.backend == 'cupy':
            if isinstance(self.v, float):
                v_init = self.v
                self.v = torch.zeros_like(x_seq[0].data)
                if v_init != 0.:
                    torch.fill_(self.v, v_init)

            spike_seq, self.v_seq = neuron_kernel.MultiStepLIFNodePTT.apply(
                x_seq.flatten(1), self.v.flatten(0), self.decay_input, self.tau, self.v_threshold, self.v_reset, self.detach_reset, self.surrogate_function.cuda_code)

            spike_seq = spike_seq.reshape(x_seq.shape)
            self.v_seq = self.v_seq.reshape(x_seq.shape)

            self.v = self.v_seq[-1].clone()

            return spike_seq

        elif self.backend == 'lava':
            if self.lava_neuron is None:
                self.lava_neuron = self.to_lava()

            spike, self.v = lava_exchange.lava_neuron_forward(self.lava_neuron, x_seq, self.v)

            return spike

        else:
            raise NotImplementedError(self.backend)



def compute_non_zero_rate(x):
    x_shape = torch.tensor(list(x.shape))
    all_neural = torch.prod(x_shape)
    z = torch.nonzero(x)
    print("After attention proj the none zero rate is", z.shape[0]/all_neural)


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0., use_imp_lif=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # self.fc1 = linear_unit(in_features, hidden_features)
        self.fc1_conv = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm2d(hidden_features)
        if not use_imp_lif:
            self.fc1_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        else:
            self.fc1_lif = MultiStepLazyStateLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        # self.fc2 = linear_unit(hidden_features, out_features)
        self.fc2_conv = nn.Conv2d(hidden_features, out_features, kernel_size=1, stride=1)
        self.fc2_bn = nn.BatchNorm2d(out_features)
        if not use_imp_lif:
            self.fc2_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        else:
            self.fc2_lif = MultiStepLazyStateLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        # self.drop = nn.Dropout(0.1)

        self.c_hidden = hidden_features
        self.c_output = out_features
    def forward(self, x):
        T,B,C,W,H = x.shape
        x = self.fc1_conv(x.flatten(0,1))
        x = self.fc1_bn(x).reshape(T,B,self.c_hidden,W,H).contiguous()
        x = self.fc1_lif(x)

        x = self.fc2_conv(x.flatten(0,1))
        x = self.fc2_bn(x).reshape(T,B,C,W,H).contiguous()
        x = self.fc2_lif(x)
        return x

class Token_QK_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, use_imp_lif=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads

        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        if not use_imp_lif:
            self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        else:
            self.q_lif = MultiStepLazyStateLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        if not use_imp_lif:
            self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        else:
            self.k_lif = MultiStepLazyStateLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        if not use_imp_lif:
            self.attn_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend='cupy')
        else:
            self.attn_lif = MultiStepLazyStateLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend='cupy')

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)
        if not use_imp_lif:
            self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        else:
            self.proj_lif = MultiStepLazyStateLIFNode(tau=2.0, detach_reset=True, backend='cupy')


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
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, use_imp_lif=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = 0.125
        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        if not use_imp_lif:
            self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        else:
            self.q_lif = MultiStepLazyStateLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        if not use_imp_lif:
            self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        else:
            self.k_lif = MultiStepLazyStateLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.v_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
        self.v_bn = nn.BatchNorm1d(dim)
        if not use_imp_lif:
            self.v_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        else:
            self.v_lif = MultiStepLazyStateLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        if not use_imp_lif:
            self.attn_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend='cupy')
        else:
            self.attn_lif = MultiStepLazyStateLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend='cupy')

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)

        if not use_imp_lif:
            self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        else:
            self.proj_lif = MultiStepLazyStateLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.qkv_mp = nn.MaxPool1d(4)

    def forward(self, x):
        T, B, C, H, W = x.shape

        x = x.flatten(3)
        T, B, C, N = x.shape
        x_for_qkv = x.flatten(0, 1)
        x_feat = x
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

class TokenSpikingTransformer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1, use_imp_lif=False):
        super().__init__()
        self.tssa = Token_QK_Attention(dim, num_heads, use_imp_lif=use_imp_lif)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features= dim, hidden_features=mlp_hidden_dim, drop=drop, use_imp_lif=use_imp_lif)

    def forward(self, x):

        x = x + self.tssa(x)
        x = x + self.mlp(x)

        return x


class SpikingTransformer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1, use_imp_lif=False):
        super().__init__()
        self.attn = Spiking_Self_Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, use_imp_lif=use_imp_lif)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop, use_imp_lif=use_imp_lif)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)

        return x

# changed on 2025-04-29
class PatchEmbedInit(nn.Module):
    def __init__(self, img_size_h=128, img_size_w=128, patch_size=4, in_channels=2, embed_dims=256,
                 recurrent_coding=False, recurrent_lif=None, pe_type=None, time_step=None,
                 use_imp_lif=False):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32  # 默认为 float32，后续可能会使用混合精度

        self.image_size = [img_size_h, img_size_w]
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.C = in_channels
        self.H, self.W = self.image_size[0] // patch_size[0], self.image_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.embed_dims = embed_dims
        # Downsampling + Res 0
        self.proj_conv = nn.Conv2d(in_channels, embed_dims // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dims // 2)
        self.proj_maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.proj1_conv = nn.Conv2d(embed_dims // 2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj1_bn = nn.BatchNorm2d(embed_dims)
        self.proj1_maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        if not use_imp_lif:
            self.proj1_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        else:
            self.proj1_lif = MultiStepLazyStateLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.proj2_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj2_bn = nn.BatchNorm2d(embed_dims)
        if not use_imp_lif:
            self.proj2_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        else:
            self.proj2_lif = MultiStepLazyStateLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.proj_res_conv = nn.Conv2d(embed_dims // 2, embed_dims, kernel_size=1, stride=2, padding=0, bias=False)
        self.proj_res_bn = nn.BatchNorm2d(embed_dims)
        if not use_imp_lif:
            self.proj_res_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        else:
            self.proj_res_lif = MultiStepLazyStateLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.recurrent_coding = recurrent_coding
        self.recurrent_lif = recurrent_lif
        self.pe_type = pe_type

        # changed on 2025-04-17
        if self.pe_type is not None:
            if self.pe_type == "3d_pe_arch_1" or self.pe_type == "3d_pe_arch_4":
                pos_embed = get_3d_sincos_pos_embed(embed_dim=self.embed_dims // 2, spatial_size=self.image_size[0] // 2,
                                                    temporal_size=time_step, output_type="pt",
                                                    )  # T HW D

                T_pe, HW_pe, D_pe = pos_embed.shape
                # pos_embed = pos_embed.to(x.dtype)
                pos_embed = pos_embed.reshape(T_pe, int(math.sqrt(HW_pe)), int(math.sqrt(HW_pe)), D_pe).unsqueeze(dim=1).permute(0, 1, 4, 2, 3).contiguous()  # T 1 C H W
                self.learnable_pos_embed = nn.Parameter(pos_embed.to(self.device, self.dtype))

            elif self.pe_type == "3d_pe_arch_2" or self.pe_type == "3d_pe_arch_3":
                pos_embed = get_sinusoid_spatial_temporal_encoding(height=img_size_h, width=img_size_w,
                                                                   time_step=time_step, d_hid=in_channels)
                
                self.learnable_pos_embed = nn.Parameter(pos_embed.to(self.device, self.dtype))  # T 1 C H W
            

        if recurrent_coding:
            if self.pe_type == "3d_pe_arch_1" or self.pe_type == "3d_pe_arch_2":
                # self.proj_temporal_conv = nn.Conv2d(embed_dims // 2, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
                # up_sampling
                self.proj_temporal_conv = nn.ConvTranspose2d(in_channels=embed_dims // 2, 
                                                             out_channels=in_channels, kernel_size=2,    # 核大小
                                                             stride=2, padding=0, output_padding=0)  # 控制输出尺寸精准匹配
                self.proj_temporal_bn = nn.BatchNorm2d(in_channels)
            elif self.pe_type == "3d_pe_arch_3" or self.pe_type == "3d_pe_arch_4" or self.pe_type == "3d_pe_arch_5":
                self.proj_temporal_conv = nn.Conv2d(embed_dims // 2, embed_dims // 2, kernel_size=3, stride=1, padding=1, bias=False)
                self.proj_temporal_bn = nn.BatchNorm2d(embed_dims // 2)
            
            if recurrent_lif is not None:
                if recurrent_lif == 'lif':
                    self.proj_temporal_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
                elif recurrent_lif == 'plif':
                    self.proj_temporal_lif = MultiStepParametricLIFNode(init_tau=2.0,detach_reset=True, backend='cupy')


    def forward(self, x):
        T, B, C, H, W = x.shape

        if not self.recurrent_coding:
            x = self.proj_conv(x.flatten(0, 1)) # TB C H W
            x = self.proj_bn(x) # TB C H W
            x = self.proj_maxpool(x).reshape(T, B, -1, H//2, W//2).contiguous()
            x = self.proj_lif(x).flatten(0, 1)  # TB C H W
        else:
            # changed on 2025-04-17
            if self.pe_type == "3d_pe_arch_2" or self.pe_type == "3d_pe_arch_3":
                x = x + self.learnable_pos_embed

            t_x = []

            if self.pe_type == "3d_pe_arch_1" or self.pe_type == "3d_pe_arch_2":
                for i in range(T):
                    if i == 0:
                        x_in = x[i] # B C H W
                    else:
                        x_in = x[i] + x_out
                        # x_in = x_out
                    x_out = self.proj_conv(x_in)    # B C H W
                    x_out = self.proj_bn(x_out)     # B C H W
                    x_out = self.proj_maxpool(x_out)    # B C H W

                    # add 3d_pe_arch_1
                    if self.pe_type == "3d_pe_arch_1":
                        # print(x_out.shape, self.learnable_pos_embed[i].shape)
                        x_out = x_out + self.learnable_pos_embed[i] # B C H W
                    
                
                    x_out = self.proj_lif(x_out.unsqueeze(0)).squeeze(0)  # B C H W
                
                    tmp = x_out
                    
                    x_out = self.proj_temporal_conv(x_out)    # B C H W
                    
                    x_out = self.proj_temporal_bn(x_out)     # B C H W
                    if self.recurrent_lif is not None:
                        x_out = self.proj_temporal_lif(x_out.unsqueeze(0)).squeeze(0) # B C H W
                
                    t_x.append(tmp)
            elif self.pe_type == "3d_pe_arch_3" or self.pe_type == "3d_pe_arch_4" or self.pe_type == "3d_pe_arch_5":
                x = self.proj_conv(x.flatten(0, 1)) # TB C H W
                x = self.proj_bn(x) # TB C H W
                x = self.proj_maxpool(x).reshape(T, B, -1, H//2, W//2).contiguous()
                if self.pe_type == "3d_pe_arch_4":
                    x = x + self.learnable_pos_embed # T B C H W

                for i in range(T):
                    if i == 0:
                        x_in = x[i] # B C H W
                    else:
                        x_in = x[i] + x_out
                        # x_in = x_out
                
                    x_out = self.proj_lif(x_in.unsqueeze(0)).squeeze(0)  # B C H W
                
                    tmp = x_out
                    # print(x_out.shape)
                    x_out = self.proj_temporal_conv(x_out)    # B C H W
                    # print(x_out.shape)
                    # assert False
                    
                    x_out = self.proj_temporal_bn(x_out)     # B C H W
                    if self.recurrent_lif is not None:
                        x_out = self.proj_temporal_lif(x_out.unsqueeze(0)).squeeze(0) # B C H W
                
                    t_x.append(tmp)

            x = torch.stack(t_x, dim=0).flatten(0, 1).contiguous() # TB C H W

        x_feat = x
        x = self.proj1_conv(x)
        x = self.proj1_bn(x)
        x = self.proj1_maxpool(x).reshape(T, B, -1, H // 4, W // 4).contiguous()
        x = self.proj1_lif(x).flatten(0, 1).contiguous()

        x = self.proj2_conv(x)
        x = self.proj2_bn(x).reshape(T, B, -1, H//4, W//4).contiguous()
        x = self.proj2_lif(x)

        x_feat = self.proj_res_conv(x_feat)
        x_feat = self.proj_res_bn(x_feat).reshape(T, B, -1, H//4, W//4).contiguous()
        x_feat = self.proj_res_lif(x_feat)

        x = x + x_feat # shortcut

        return x

class PatchEmbeddingStage(nn.Module):
    def __init__(self, img_size_h=128, img_size_w=128, patch_size=4, in_channels=2, embed_dims=256, use_imp_lif=False):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.C = in_channels
        self.H, self.W = self.image_size[0] // patch_size[0], self.image_size[1] // patch_size[1]
        self.num_patches = self.H * self.W

        self.proj3_conv = nn.Conv2d(embed_dims//2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj3_bn = nn.BatchNorm2d(embed_dims)
        self.proj3_maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        if not use_imp_lif:
            self.proj3_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        else:
            self.proj3_lif = MultiStepLazyStateLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.proj4_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj4_bn = nn.BatchNorm2d(embed_dims)
        if not use_imp_lif:
            self.proj4_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        else:
            self.proj4_lif = MultiStepLazyStateLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.proj_res_conv = nn.Conv2d(embed_dims//2, embed_dims, kernel_size=1, stride=2, padding=0, bias=False)
        self.proj_res_bn = nn.BatchNorm2d(embed_dims)
        if not use_imp_lif:
            self.proj_res_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        else:
            self.proj_res_lif = MultiStepLazyStateLIFNode(tau=2.0, detach_reset=True, backend='cupy')

    def forward(self, x):
        T, B, C, H, W = x.shape
        # Downsampling + Res

        x = x.flatten(0, 1).contiguous()
        x_feat = x

        x = self.proj3_conv(x)
        x = self.proj3_bn(x)
        x = self.proj3_maxpool(x).reshape(T, B, -1, H//2, W//2).contiguous()
        x = self.proj3_lif(x).flatten(0, 1).contiguous()

        x = self.proj4_conv(x)
        x = self.proj4_bn(x).reshape(T, B, -1, H//2, W//2).contiguous()
        x = self.proj4_lif(x)

        x_feat = self.proj_res_conv(x_feat)
        x_feat = self.proj_res_bn(x_feat).reshape(T, B, -1, H//2, W//2).contiguous()
        x_feat = self.proj_res_lif(x_feat)

        x = x + x_feat # shortcut

        return x

# changed on 2025-04-29
class hierarchical_spiking_transformer(nn.Module):
    def __init__(self,
                 T=4, recurrent_coding=False, recurrent_lif=None, pe_type=None, 
                 img_size_h=128, img_size_w=128, use_imp_lif=False, patch_size=16, in_channels=2, num_classes=11,
                 embed_dims=[64, 128, 256], num_heads=[1, 2, 4], mlp_ratios=[4, 4, 4], qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[6, 8, 6], sr_ratios=[8, 4, 2],
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.T = T
        self.recurrent_coding = recurrent_coding
        self.recurrent_lif = recurrent_lif
        self.pe_type = pe_type
        self.use_imp_lif = use_imp_lif

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule

        patch_embed1 = PatchEmbedInit(img_size_h=img_size_h,
                                 img_size_w=img_size_w,
                                 patch_size=patch_size,
                                 in_channels=in_channels,
                                 embed_dims=embed_dims // 4,
                                 recurrent_coding=recurrent_coding, recurrent_lif=recurrent_lif, pe_type=pe_type,
                                 time_step=T, use_imp_lif=use_imp_lif
                                 )

        stage1 = nn.ModuleList([TokenSpikingTransformer(
            dim=embed_dims // 4, num_heads=num_heads, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
            norm_layer=norm_layer, sr_ratio=sr_ratios, use_imp_lif=use_imp_lif)
            for j in range(1)])

        patch_embed2 = PatchEmbeddingStage(img_size_h=img_size_h,
                                       img_size_w=img_size_w,
                                       patch_size=patch_size,
                                       in_channels=in_channels,
                                       embed_dims=embed_dims // 2, use_imp_lif=use_imp_lif)


        stage2 = nn.ModuleList([TokenSpikingTransformer(
            dim=embed_dims // 2, num_heads=num_heads, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
            norm_layer=norm_layer, sr_ratio=sr_ratios, use_imp_lif=use_imp_lif)
            for j in range(2)])


        patch_embed3 = PatchEmbeddingStage(img_size_h=img_size_h,
                                       img_size_w=img_size_w,
                                       patch_size=patch_size,
                                       in_channels=in_channels,
                                       embed_dims=embed_dims, use_imp_lif=use_imp_lif)

        stage3 = nn.ModuleList([SpikingTransformer(
            dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
            norm_layer=norm_layer, sr_ratio=sr_ratios, use_imp_lif=use_imp_lif)
            for j in range(depths - 3)])

        setattr(self, f"patch_embed1", patch_embed1)
        setattr(self, f"patch_embed2", patch_embed2)
        setattr(self, f"patch_embed3", patch_embed3)
        setattr(self, f"stage1", stage1)
        setattr(self, f"stage2", stage2)
        setattr(self, f"stage3", stage3)

        # classification head 这里不需要脉冲，因为输入的是在T时长平均发射值
        self.head = nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    @torch.jit.ignore
    def _get_pos_embed(self, pos_embed, patch_embed3, H, W):
        if H * W == self.patch_embed3.num_patches:
            return pos_embed
        else:
            return F.interpolate(
                pos_embed.reshape(1, patch_embed3.H, patch_embed3.W, -1).permute(0, 3, 1, 2),
                size=(H, W), mode="bilinear").reshape(1, -1, H * W).permute(0, 2, 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):

        stage1 = getattr(self, f"stage1")
        stage2 = getattr(self, f"stage2")
        stage3 = getattr(self, f"stage3")
        patch_embed1 = getattr(self, f"patch_embed1")
        patch_embed2 = getattr(self, f"patch_embed2")
        patch_embed3 = getattr(self, f"patch_embed3")

        x = patch_embed1(x)
        for blk in stage1:
            x = blk(x)

        x = patch_embed2(x)
        for blk in stage2:
            x = blk(x)

        x = patch_embed3(x)
        for blk in stage3:
            x = blk(x)

        return x.flatten(3).mean(3)

    def forward(self, x):
        T = self.T
        x = (x.unsqueeze(0)).repeat(T, 1, 1, 1, 1)
        x = self.forward_features(x)
        x = self.head(x.mean(0))
        return x

def QKFormer_10_384(T=1, **kwargs):
    model = hierarchical_spiking_transformer(
        T=T,
        img_size_h=224, img_size_w=224,
        patch_size=16, embed_dims=384, num_heads=6, mlp_ratios=4,
        in_channels=3, num_classes=1000, qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=10, sr_ratios=1,
        **kwargs
    )
    return model

def QKFormer_10_512(T=1, **kwargs):
    model = hierarchical_spiking_transformer(
        T=T,
        img_size_h=224, img_size_w=224,
        patch_size=16, embed_dims=512, num_heads=8, mlp_ratios=4,
        in_channels=3, num_classes=1000, qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=10, sr_ratios=1,
        **kwargs
    )
    return model

# changed on 2025-04-29
def QKFormer_10_768(T=1, recurrent_coding=False, recurrent_lif=None, pe_type=None, finetune = None, 
                    input_H=None, input_W=None, use_imp_lif=False, **kwargs):
    model = hierarchical_spiking_transformer(
        T=T, recurrent_coding=recurrent_coding, 
        recurrent_lif=recurrent_lif, pe_type=pe_type,
        img_size_h=input_H, img_size_w=input_W, use_imp_lif=use_imp_lif, 
        patch_size=16, embed_dims=768, num_heads=12, mlp_ratios=4,
        in_channels=3, num_classes=1000, qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=10, sr_ratios=1,
        **kwargs
    )
    print(f"using recurrent coding: {model.recurrent_coding}")
    print(f"using recurrent LIF: {model.recurrent_lif}")
    print(f"postion embedding methods: {model.pe_type}")
    print(f"using imp lif: {model.use_imp_lif}")

    if finetune is not None:
        
        checkpoint = torch.load(finetune)
        pretrain_dict = checkpoint['model']
        model_dict = model.state_dict()

        pretrain_dict = {k:v for k,v in pretrain_dict.items() if k in model_dict and k != 'head.weight' and k != 'head.bias'}
        # pretrain_dict = {k:v for k,v in pretrain_dict.items() if k in model_dict}

        backbone_dict = {k:v for k,v in model_dict.items() if k not in pretrain_dict}


        model_dict.update(pretrain_dict)

        model.load_state_dict(model_dict)

        for name,param in model.named_parameters():
            if name in pretrain_dict  :
                
                print(f"{name} loaded from pretrained model")
                # param.requires_grad = False
              
            if name in backbone_dict:
                print(f"{name} random initialised and also can be trained")
                continue
    
        return model

    return model


if __name__ == '__main__':
    H = 128
    W = 128
    x = torch.randn(2, 3, 224, 224).cuda()
    model = QKFormer_10_768(T = 4).cuda()

    model.eval()
    from torchinfo import summary
    summary(model, input_size=(1, 3, 224, 224))
