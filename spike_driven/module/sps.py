import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import (
    MultiStepLIFNode,
    MultiStepParametricLIFNode,
)
from timm.models.layers import to_2tuple
import math

from spikingjelly.clock_driven import surrogate, lava_exchange
from spikingjelly.clock_driven import neuron_kernel, cu_kernel_opt
from typing import Callable

from .embeddings import get_3d_sincos_pos_embed, get_sinusoid_spatial_temporal_encoding

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


class MS_SPS(nn.Module):
    def __init__(
        self,
        img_size_h=128,
        img_size_w=128,
        patch_size=4,
        in_channels=2,
        embed_dims=256,
        pooling_stat="1111",
        spike_mode="lif",
        recurrent_coding=False,     # changed on 2025-04-13
        recurrent_lif=None,         # changed on 2025-04-13
        pe_type=None,               # changed on 2025-04-17
        time_step=None,             # changed on 2025-04-17
        temporal_conv_type=None,    # changed on 2025-09-22
        # use_diversity_loss=False,   # changed on 2025-04-23
        # lif_recurrent_state=None,   # changed on 2025-04-27
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32  # 默认为 float32，后续可能会使用混合精度

        self.image_size = [img_size_h, img_size_w]
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.pooling_stat = pooling_stat
        self.embed_dims = embed_dims
        # self.lif_recurrent_state = lif_recurrent_state

        self.C = in_channels
        self.H, self.W = (
            self.image_size[0] // patch_size[0],
            self.image_size[1] // patch_size[1],
        )
        self.num_patches = self.H * self.W
        self.proj_conv = nn.Conv2d(
            in_channels, embed_dims // 8, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.proj_bn = nn.BatchNorm2d(embed_dims // 8)
        if spike_mode == "lif":
            self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        elif spike_mode == "plif":
            self.proj_lif = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy"
            )
        self.maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )

        self.proj_conv1 = nn.Conv2d(
            embed_dims // 8,
            embed_dims // 4,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.proj_bn1 = nn.BatchNorm2d(embed_dims // 4)
        if spike_mode == "lif":
            self.proj_lif1 = MultiStepLIFNode(
                tau=2.0, detach_reset=True, backend="cupy"
            )
            # self.proj_lif1 = MultiStepLazyStateLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        elif spike_mode == "plif":
            self.proj_lif1 = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy"
            )
        self.maxpool1 = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )

        self.proj_conv2 = nn.Conv2d(
            embed_dims // 4,
            embed_dims // 2,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.proj_bn2 = nn.BatchNorm2d(embed_dims // 2)
        if spike_mode == "lif":
            self.proj_lif2 = MultiStepLIFNode(
                tau=2.0, detach_reset=True, backend="cupy"
            )
            # self.proj_lif2 = MultiStepLazyStateLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        elif spike_mode == "plif":
            self.proj_lif2 = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy"
            )
        self.maxpool2 = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )

        self.proj_conv3 = nn.Conv2d(
            embed_dims // 2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.proj_bn3 = nn.BatchNorm2d(embed_dims)
        if spike_mode == "lif":
            self.proj_lif3 = MultiStepLIFNode(
                tau=2.0, detach_reset=True, backend="cupy"
            )
            # self.proj_lif3 = MultiStepLazyStateLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        elif spike_mode == "plif":
            self.proj_lif3 = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy"
            )
        self.maxpool3 = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )

        self.rpe_conv = nn.Conv2d(
            embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.rpe_bn = nn.BatchNorm2d(embed_dims)
        if spike_mode == "lif":
            self.rpe_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
            # self.rpe_lif = MultiStepLazyStateLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        elif spike_mode == "plif":
            self.rpe_lif = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy"
            )

        # changed on 2025-04-13
        self.recurrent_coding = recurrent_coding
        self.recurrent_lif = recurrent_lif
        self.pe_type = pe_type
        self.temporal_conv_type=temporal_conv_type
        # self.use_diversity_loss = use_diversity_loss

        # changed on 2025-04-17
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

    def forward(self, x, hook=None):
        T, B, _, H, W = x.shape
        ratio = 1
        # changed on 2025-04-13
        if not self.recurrent_coding:
            x = self.proj_conv(x.flatten(0, 1))  # have some fire value
            x = self.proj_bn(x).reshape(T, B, -1, H // ratio, W // ratio).contiguous()
            x = self.proj_lif(x) # T B C H W

        else:
            t_x = []

            if self.pe_type == "stf_1":
                for i in range(T):
                    if i == 0:
                        x_in = x[i] # B C H W
                    else:
                        x_in = x[i] + x_out
                        # x_in = x_out
                    x_out = self.proj_conv(x_in)    # B C H W
                    x_out = self.proj_bn(x_out)     # B C H W
                    x_out = self.proj_lif(x_out.unsqueeze(0)).squeeze(0)  # B C H W
                
                    tmp = x_out
                    
                    if self.temporal_conv_type == "conv1d":
                        x_out = x_out.flatten(2) # B C N
                        x_out = self.proj_temporal_conv(x_out)  # B C N
                        x_out = self.proj_temporal_bn(x_out).reshape(B, -1, H, W).contiguous()

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

            x = torch.stack(t_x, dim=0) # T B C H W
        # changed on 2025-04-24
        # if self.use_diversity_loss:
        #     diversity_coding = x

        if hook is not None:
            hook[self._get_name() + "_lif"] = x.detach()
        x = x.flatten(0, 1).contiguous()
        if self.pooling_stat[0] == "1":
            x = self.maxpool(x)
            ratio *= 2

        x = self.proj_conv1(x)
        x = self.proj_bn1(x).reshape(T, B, -1, H // ratio, W // ratio).contiguous()
        # changed on 2025-04-27
        # if self.lif_recurrent_state is not None and self.lif_recurrent_state[0] == "1":
        #     tmp_x = []
        #     for t in range(T):
        #         if t == 0:
        #             x_in = x[t]    # B C H W
        #         else:
        #             x_in = x[t] + x_out
        #         x_out = self.proj_lif1(x_in.unsqueeze(0)).squeeze(0)   # B C H W
        #         tmp_x.append(x_out)
            
        #     x = torch.stack(tmp_x, dim=0)    # T B C H W
        # else:
        x = self.proj_lif1(x)

        if hook is not None:
            hook[self._get_name() + "_lif1"] = x.detach()
        x = x.flatten(0, 1).contiguous()
        if self.pooling_stat[1] == "1":
            x = self.maxpool1(x)
            ratio *= 2

        x = self.proj_conv2(x)
        x = self.proj_bn2(x).reshape(T, B, -1, H // ratio, W // ratio).contiguous()
        # changed on 2025-04-27
        # if self.lif_recurrent_state is not None and self.lif_recurrent_state[1] == "1":
        #     tmp_x = []
        #     for t in range(T):
        #         if t == 0:
        #             x_in = x[t]    # B C H W
        #         else:
        #             x_in = x[t] + x_out
        #         x_out = self.proj_lif2(x_in.unsqueeze(0)).squeeze(0)   # B C H W
        #         tmp_x.append(x_out)
        #     x = torch.stack(tmp_x, dim=0)    # T B C H W
        # else:
        x = self.proj_lif2(x)
        
        if hook is not None:
            hook[self._get_name() + "_lif2"] = x.detach()
        x = x.flatten(0, 1).contiguous()
        if self.pooling_stat[2] == "1":
            x = self.maxpool2(x)
            ratio *= 2

        x = self.proj_conv3(x)
        x = self.proj_bn3(x)
        if self.pooling_stat[3] == "1":
            x = self.maxpool3(x)
            ratio *= 2

        x_feat = x
        # changed on 2025-04-27
        # if self.lif_recurrent_state is not None and self.lif_recurrent_state[2] == "1":
        #     tmp_x = []
        #     x = x.reshape(T, B, -1, H // ratio, W // ratio).contiguous()  # T B C H W
        #     for t in range(T):
        #         if t == 0:
        #             x_in = x[t]    # B C H W
        #         else:
        #             x_in = x[t] + x_out
        #         x_out = self.proj_lif3(x_in.unsqueeze(0)).squeeze(0)   # B C H W
        #         tmp_x.append(x_out)
        #     x = torch.stack(tmp_x, dim=0)    # T B C H W
        # else:
        x = self.proj_lif3(x.reshape(T, B, -1, H // ratio, W // ratio).contiguous())
        
        if hook is not None:
            hook[self._get_name() + "_lif3"] = x.detach()
        x = x.flatten(0, 1).contiguous()
        x = self.rpe_conv(x)
        x = self.rpe_bn(x)
        x = (x + x_feat).reshape(T, B, -1, H // ratio, W // ratio).contiguous()

        H, W = H // self.patch_size[0], W // self.patch_size[1]
        # changed on 2025-04-24
        # if self.use_diversity_loss:
        #     return x, (H, W), hook, diversity_coding
        # else:
        return x, (H, W), hook



class MS_SPS_Maxpooling_LIF_changed(nn.Module):
    def __init__(
        self,
        img_size_h=128,
        img_size_w=128,
        patch_size=4,
        in_channels=2,
        embed_dims=256,
        pooling_stat="1111",
        spike_mode="lif",
        recurrent_coding=False,     # changed on 2025-04-13
        recurrent_lif=None,         # changed on 2025-04-13
        pe_type=None,               # changed on 2025-04-17
        time_step=None,             # changed on 2025-04-17
        temporal_conv_type=None,    # changed on 2025-09-22
        # use_diversity_loss=False,   # changed on 2025-04-23
        # lif_recurrent_state=None,   # changed on 2025-04-27
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32  # 默认为 float32，后续可能会使用混合精度

        self.image_size = [img_size_h, img_size_w]
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.pooling_stat = pooling_stat
        self.embed_dims = embed_dims
        # self.lif_recurrent_state = lif_recurrent_state

        self.C = in_channels
        self.H, self.W = (
            self.image_size[0] // patch_size[0],
            self.image_size[1] // patch_size[1],
        )
        self.num_patches = self.H * self.W
        self.proj_conv = nn.Conv2d(
            in_channels, embed_dims // 8, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.proj_bn = nn.BatchNorm2d(embed_dims // 8)
        if spike_mode == "lif":
            self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        elif spike_mode == "plif":
            self.proj_lif = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy"
            )
        self.maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )

        self.proj_conv1 = nn.Conv2d(
            embed_dims // 8,
            embed_dims // 4,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.proj_bn1 = nn.BatchNorm2d(embed_dims // 4)
        if spike_mode == "lif":
            self.proj_lif1 = MultiStepLIFNode(
                tau=2.0, detach_reset=True, backend="cupy"
            )
            # self.proj_lif1 = MultiStepLazyStateLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        elif spike_mode == "plif":
            self.proj_lif1 = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy"
            )
        self.maxpool1 = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )

        self.proj_conv2 = nn.Conv2d(
            embed_dims // 4,
            embed_dims // 2,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.proj_bn2 = nn.BatchNorm2d(embed_dims // 2)
        if spike_mode == "lif":
            self.proj_lif2 = MultiStepLIFNode(
                tau=2.0, detach_reset=True, backend="cupy"
            )
            # self.proj_lif2 = MultiStepLazyStateLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        elif spike_mode == "plif":
            self.proj_lif2 = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy"
            )
        self.maxpool2 = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )

        self.proj_conv3 = nn.Conv2d(
            embed_dims // 2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.proj_bn3 = nn.BatchNorm2d(embed_dims)
        if spike_mode == "lif":
            self.proj_lif3 = MultiStepLIFNode(
                tau=2.0, detach_reset=True, backend="cupy"
            )
            # self.proj_lif3 = MultiStepLazyStateLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        elif spike_mode == "plif":
            self.proj_lif3 = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy"
            )
        self.maxpool3 = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )

        self.rpe_conv = nn.Conv2d(
            embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.rpe_bn = nn.BatchNorm2d(embed_dims)
        if spike_mode == "lif":
            self.rpe_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
            # self.rpe_lif = MultiStepLazyStateLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        elif spike_mode == "plif":
            self.rpe_lif = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy"
            )

        # changed on 2025-04-13
        self.recurrent_coding = recurrent_coding
        self.recurrent_lif = recurrent_lif
        self.pe_type = pe_type
        self.temporal_conv_type=temporal_conv_type
        # self.use_diversity_loss = use_diversity_loss

        # changed on 2025-04-17
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

    def forward(self, x, hook=None):
        T, B, _, H, W = x.shape
        ratio = 1
        # changed on 2025-04-13
        if not self.recurrent_coding:
            x = self.proj_conv(x.flatten(0, 1))  # have some fire value
            x = self.proj_bn(x).reshape(T, B, -1, H // ratio, W // ratio).contiguous()
            x = x.flatten(0, 1).contiguous() # TB C H W
            if self.pooling_stat[0] == "1":
                x = self.maxpool(x)
                ratio *= 2
                x = x.reshape(T, B, -1, H // ratio, W // ratio).contiguous()    # T B C H W
            elif self.pooling_stat[0] == "0":
                x = x.reshape(T, B, -1, H // ratio, W // ratio).contiguous()    # T B C H W
            x = self.proj_lif(x) # T B C H W
            
        else:
            t_x = []

            if self.pe_type == "stf_1":
                # # 不会导致for loop导致ratio不断增大
                # if self.pooling_stat[0] == "1":
                #     ratio *= 2

                for i in range(T):
                    if i == 0:
                        x_in = x[i] # B C H W
                    else:
                        x_in = x[i] + x_out
                        # x_in = x_out
                    x_out = self.proj_conv(x_in)    # B C H W
                    x_out = self.proj_bn(x_out)     # B C H W

                    # if self.pooling_stat[0] == "1":
                    #     x_out = self.maxpool(x_out)
                    #     x_out = x_out.reshape(B, -1, H // ratio, W // ratio).contiguous()    # B C H W

                    x_out = self.proj_lif(x_out.unsqueeze(0)).squeeze(0)  # B C H W
                
                    tmp = x_out
                    
                    if self.temporal_conv_type == "conv1d":
                        x_out = x_out.flatten(2) # B C N
                        x_out = self.proj_temporal_conv(x_out)  # B C N
                        x_out = self.proj_temporal_bn(x_out).reshape(B, -1, H, W).contiguous()

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

            x = torch.stack(t_x, dim=0) # T B C H W
        # changed on 2025-04-24
        # if self.use_diversity_loss:
        #     diversity_coding = x

        if hook is not None:
            hook[self._get_name() + "_lif"] = x.detach()
        x = x.flatten(0, 1).contiguous()
        # if self.pooling_stat[0] == "1":
        #     x = self.maxpool(x)
        #     ratio *= 2

        x = self.proj_conv1(x)
        x = self.proj_bn1(x).reshape(T, B, -1, H // ratio, W // ratio).contiguous()
        # changed on 2025-04-27
        # if self.lif_recurrent_state is not None and self.lif_recurrent_state[0] == "1":
        #     tmp_x = []
        #     for t in range(T):
        #         if t == 0:
        #             x_in = x[t]    # B C H W
        #         else:
        #             x_in = x[t] + x_out
        #         x_out = self.proj_lif1(x_in.unsqueeze(0)).squeeze(0)   # B C H W
        #         tmp_x.append(x_out)
            
        #     x = torch.stack(tmp_x, dim=0)    # T B C H W
        # else:

        # changed on 2025-05-25
        x = x.flatten(0, 1).contiguous()
        if self.pooling_stat[1] == "1":
            x = self.maxpool1(x)
            ratio *= 2
            x = x.reshape(T, B, -1, H // ratio, W // ratio).contiguous()  # T B C H W
        elif self.pooling_stat[1] == "0":
            x = x.reshape(T, B, -1, H // ratio, W // ratio).contiguous()  # T B C H W

        x = self.proj_lif1(x)

        if hook is not None:
            hook[self._get_name() + "_lif1"] = x.detach()
        x = x.flatten(0, 1).contiguous()
        # if self.pooling_stat[1] == "1":
        #     x = self.maxpool1(x)
        #     ratio *= 2

        x = self.proj_conv2(x)
        x = self.proj_bn2(x).reshape(T, B, -1, H // ratio, W // ratio).contiguous()
        # changed on 2025-04-27
        # if self.lif_recurrent_state is not None and self.lif_recurrent_state[1] == "1":
        #     tmp_x = []
        #     for t in range(T):
        #         if t == 0:
        #             x_in = x[t]    # B C H W
        #         else:
        #             x_in = x[t] + x_out
        #         x_out = self.proj_lif2(x_in.unsqueeze(0)).squeeze(0)   # B C H W
        #         tmp_x.append(x_out)
        #     x = torch.stack(tmp_x, dim=0)    # T B C H W
        # else:

        # changed on 2025-05-25
        x = x.flatten(0, 1).contiguous()
        if self.pooling_stat[2] == "1":
            x = self.maxpool2(x)
            ratio *= 2
            x = x.reshape(T, B, -1, H // ratio, W // ratio).contiguous()  # T B C H W
        elif self.pooling_stat[2] == "0":
            x = x.reshape(T, B, -1, H // ratio, W // ratio).contiguous()  # T B C H W
        
        x = self.proj_lif2(x)
        
        if hook is not None:
            hook[self._get_name() + "_lif2"] = x.detach()
        x = x.flatten(0, 1).contiguous()
        # if self.pooling_stat[2] == "1":
        #     x = self.maxpool2(x)
        #     ratio *= 2


        x = self.proj_conv3(x)
        x = self.proj_bn3(x)
        if self.pooling_stat[3] == "1":
            x = self.maxpool3(x)
            ratio *= 2

        x_feat = x
        # changed on 2025-04-27
        # if self.lif_recurrent_state is not None and self.lif_recurrent_state[2] == "1":
        #     tmp_x = []
        #     x = x.reshape(T, B, -1, H // ratio, W // ratio).contiguous()  # T B C H W
        #     for t in range(T):
        #         if t == 0:
        #             x_in = x[t]    # B C H W
        #         else:
        #             x_in = x[t] + x_out
        #         x_out = self.proj_lif3(x_in.unsqueeze(0)).squeeze(0)   # B C H W
        #         tmp_x.append(x_out)
        #     x = torch.stack(tmp_x, dim=0)    # T B C H W
        # else:
        x = self.proj_lif3(x.reshape(T, B, -1, H // ratio, W // ratio).contiguous())
        
        if hook is not None:
            hook[self._get_name() + "_lif3"] = x.detach()
        x = x.flatten(0, 1).contiguous()
        x = self.rpe_conv(x)
        x = self.rpe_bn(x)
        x = (x + x_feat).reshape(T, B, -1, H // ratio, W // ratio).contiguous()

        H, W = H // self.patch_size[0], W // self.patch_size[1]
        # changed on 2025-04-24
        # if self.use_diversity_loss:
        #     return x, (H, W), hook, diversity_coding
        # else:
        
        return x, (H, W), hook


