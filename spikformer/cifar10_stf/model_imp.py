import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepLIFNode, MultiStepParametricLIFNode
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import torch.nn.functional as F
from functools import partial
import math

from embeddings import get_3d_sincos_pos_embed, get_sinusoid_spatial_temporal_encoding


__all__ = ['spikformer_IMP']

from spikingjelly.clock_driven import surrogate, lava_exchange
from spikingjelly.clock_driven import neuron_kernel, cu_kernel_opt
from typing import Callable
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



class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1_linear = nn.Linear(in_features, hidden_features)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        # self.fc1_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.fc1_lif = MultiStepLazyStateLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.fc2_linear = nn.Linear(hidden_features, out_features)
        self.fc2_bn = nn.BatchNorm1d(out_features)
        # self.fc2_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.fc2_lif = MultiStepLazyStateLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        T,B,N,C = x.shape
        x_ = x.flatten(0, 1)
        x = self.fc1_linear(x_)
        x = self.fc1_bn(x.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, self.c_hidden).contiguous()
        x = self.fc1_lif(x)

        x = self.fc2_linear(x.flatten(0,1))
        x = self.fc2_bn(x.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C).contiguous()
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
        # self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.q_lif = MultiStepLazyStateLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.k_linear = nn.Linear(dim, dim)
        self.k_bn = nn.BatchNorm1d(dim)
        # self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.k_lif = MultiStepLazyStateLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.v_linear = nn.Linear(dim, dim)
        self.v_bn = nn.BatchNorm1d(dim)
        # self.v_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.v_lif = MultiStepLazyStateLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        # self.attn_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend='cupy')
        self.attn_lif = MultiStepLazyStateLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend='cupy')

        self.proj_linear = nn.Linear(dim, dim)
        self.proj_bn = nn.BatchNorm1d(dim)
        # self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.proj_lif = MultiStepLazyStateLIFNode(tau=2.0, detach_reset=True, backend='cupy')

    def forward(self, x):
        T,B,N,C = x.shape

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
        x = self.proj_lif(self.proj_bn(self.proj_linear(x).transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C))
        return x

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
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class SPS(nn.Module):
    def __init__(self, img_size_h=128, img_size_w=128, patch_size=4, in_channels=2, embed_dims=256, 
                 recurrent_coding=False, recurrent_lif=None, time_step=None, pe_type=None):          # changed on 2025-04-13
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
        # self.proj_lif1 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.proj_lif1 = MultiStepLazyStateLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.proj_conv2 = nn.Conv2d(embed_dims//4, embed_dims//2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn2 = nn.BatchNorm2d(embed_dims//2)
        # self.proj_lif2 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.proj_lif2 = MultiStepLazyStateLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj_conv3 = nn.Conv2d(embed_dims//2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn3 = nn.BatchNorm2d(embed_dims)
        # self.proj_lif3 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.proj_lif3 = MultiStepLazyStateLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.maxpool3 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.rpe_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.rpe_bn = nn.BatchNorm2d(embed_dims)
        # self.rpe_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.rpe_lif = MultiStepLazyStateLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        # changed on 2025-04-13
        self.recurrent_coding = recurrent_coding
        self.recurrent_lif = recurrent_lif
        self.pe_type = pe_type

        if self.pe_type is not None:
            if self.pe_type == "3d_pe_arch_1" or self.pe_type == "3d_pe_arch_4":
                pos_embed = get_3d_sincos_pos_embed(embed_dim=self.embed_dims // 8, spatial_size=self.image_size[0],
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
            self.proj_temporal_conv = nn.Conv2d(embed_dims // 8, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.proj_temporal_bn = nn.BatchNorm2d(in_channels)
            
            if recurrent_lif is not None:
                if recurrent_lif == 'lif':
                    self.proj_temporal_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
                elif recurrent_lif == 'plif':
                    self.proj_temporal_lif = MultiStepParametricLIFNode(init_tau=2.0,detach_reset=True, backend='cupy')
        

    def forward(self, x):
        T, B, C, H, W = x.shape

        # changed on 2025-04-13
        if not self.recurrent_coding:
            x = self.proj_conv(x.flatten(0, 1)) # have some fire value
            x = self.proj_bn(x).reshape(T, B, -1, H, W).contiguous()
            x = self.proj_lif(x).flatten(0, 1).contiguous()
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

                    # add 3d_pe_arch_1
                    if self.pe_type == "3d_pe_arch_1":
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
                x = self.proj_bn(x).reshape(T, B, -1, H, W) # T B C H W
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

        x = x.flatten(-2).transpose(-1, -2)  # T,B,N,C
        return x


class Spikformer(nn.Module):
    def __init__(self,
                 img_size_h=128, img_size_w=128, patch_size=16, in_channels=2, num_classes=11,
                 embed_dims=[64, 128, 256], num_heads=[1, 2, 4], mlp_ratios=[4, 4, 4], qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[6, 8, 6], sr_ratios=[8, 4, 2], T = 4, recurrent_coding=False, recurrent_lif=None, # changed on 2025-04-13,
                 pe_type=None, 
                 ):
        super().__init__()
        self.T = T  # time step
        self.num_classes = num_classes
        self.depths = depths
        self.recurrent_coding = recurrent_coding
        self.pe_type = pe_type
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule

        patch_embed = SPS(img_size_h=img_size_h,
                                 img_size_w=img_size_w,
                                 patch_size=patch_size,
                                 in_channels=in_channels,
                                 embed_dims=embed_dims,
                                 recurrent_coding=recurrent_coding,
                                 recurrent_lif=recurrent_lif,
                                 time_step=T,
                                 pe_type=pe_type,
                                 )                # changed on 2025-04-13

        block = nn.ModuleList([Block(
            dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
            norm_layer=norm_layer, sr_ratio=sr_ratios)
            for j in range(depths)])

        setattr(self, f"patch_embed", patch_embed)
        setattr(self, f"block", block)

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

    def forward_features(self, x):

        block = getattr(self, f"block")
        patch_embed = getattr(self, f"patch_embed")

        x = patch_embed(x)
        for blk in block:
            x = blk(x)
        return x.mean(2)

    def forward(self, x):
        x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
        x = self.forward_features(x)
        x = self.head(x.mean(0))
        return x


@register_model
def spikformer_IMP(pretrained=False, **kwargs):
    model = Spikformer(
        # img_size_h=224, img_size_w=224,
        # patch_size=16, embed_dims=768, num_heads=12, mlp_ratios=4,
        # in_channels=3, num_classes=1000, qkv_bias=False,
        # norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=12, sr_ratios=1,
        **kwargs
    )
    print(f"using recurrent code: {model.recurrent_coding}")
    print(f"using pe_type: {model.pe_type}")
    print(f"using IMP")
    model.default_cfg = _cfg()
    return model