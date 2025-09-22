import torch.nn as nn
from timm.models.layers import DropPath
from spikingjelly.clock_driven.neuron import (
    MultiStepLIFNode,
    MultiStepParametricLIFNode,
)
import torch

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

class Erode(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.MaxPool3d(
            kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)
        )

    def forward(self, x):
        return self.pool(x)


class MS_MLP_Conv(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        drop=0.0,
        spike_mode="lif",
        layer=0,
        # lif_recurrent_state=None, # changed on 2025-04-27
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.res = in_features == hidden_features
        self.fc1_conv = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm2d(hidden_features)
        if spike_mode == "lif":
            self.fc1_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
            # self.fc1_lif = MultiStepLazyStateLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        elif spike_mode == "plif":
            self.fc1_lif = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy"
            )

        self.fc2_conv = nn.Conv2d(
            hidden_features, out_features, kernel_size=1, stride=1
        )
        self.fc2_bn = nn.BatchNorm2d(out_features)
        if spike_mode == "lif":
            self.fc2_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
            # self.fc2_lif = MultiStepLazyStateLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        elif spike_mode == "plif":
            self.fc2_lif = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy"
            )

        self.c_hidden = hidden_features
        self.c_output = out_features
        self.layer = layer
        # self.lif_recurrent_state = lif_recurrent_state

    def forward(self, x, hook=None):
        T, B, C, H, W = x.shape
        identity = x
        # changed on 2025-04-27
        # if self.lif_recurrent_state is not None and self.lif_recurrent_state[0] == "1":
        #     tmp_x = []
        #     for t in range(T):
        #         if t == 0:
        #             x_in = x[t]    # B C H W
        #         else:
        #             x_in = x[t] + x_out
        #         x_out = self.fc1_lif(x_in.unsqueeze(0)).squeeze(0)   # B C H W
        #         tmp_x.append(x_out)
            
        #     x = torch.stack(tmp_x, dim=0)    # T B C H W
        # else:
        x = self.fc1_lif(x)
        
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_fc1_lif"] = x.detach()
        x = self.fc1_conv(x.flatten(0, 1))
        x = self.fc1_bn(x).reshape(T, B, self.c_hidden, H, W).contiguous()
        if self.res:
            x = identity + x
            identity = x
        # changed on 2025-04-27
        # if self.lif_recurrent_state is not None and self.lif_recurrent_state[1] == "1":
        #     tmp_x = []
        #     for t in range(T):
        #         if t == 0:
        #             x_in = x[t]    # B C H W
        #         else:
        #             x_in = x[t] + x_out
        #         x_out = self.fc2_lif(x_in.unsqueeze(0)).squeeze(0)   # B C H W
        #         tmp_x.append(x_out)
            
        #     x = torch.stack(tmp_x, dim=0)    # T B C H W
        # else:
        x = self.fc2_lif(x)
        
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_fc2_lif"] = x.detach()
        x = self.fc2_conv(x.flatten(0, 1))
        x = self.fc2_bn(x).reshape(T, B, C, H, W).contiguous()

        x = x + identity
        return x, hook


class MS_SSA_Conv(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        sr_ratio=1,
        mode="direct_xor",
        spike_mode="lif",
        dvs=False,
        layer=0,
        # lif_recurrent_state=None, # changed on 2025-04-27
    ):
        super().__init__()
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.dvs = dvs
        self.num_heads = num_heads
        if dvs:
            self.pool = Erode()
        self.scale = 0.125
        self.q_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm2d(dim)
        if spike_mode == "lif":
            self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
            # self.q_lif = MultiStepLazyStateLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        elif spike_mode == "plif":
            self.q_lif = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy"
            )

        self.k_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm2d(dim)
        if spike_mode == "lif":
            self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
            # self.k_lif = MultiStepLazyStateLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        elif spike_mode == "plif":
            self.k_lif = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy"
            )

        self.v_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.v_bn = nn.BatchNorm2d(dim)
        if spike_mode == "lif":
            self.v_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
            # self.v_lif = MultiStepLazyStateLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        elif spike_mode == "plif":
            self.v_lif = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy"
            )

        if spike_mode == "lif":
            self.attn_lif = MultiStepLIFNode(
                tau=2.0, v_threshold=0.5, detach_reset=True, backend="cupy"
            )
            # self.attn_lif = MultiStepLazyStateLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        elif spike_mode == "plif":
            self.attn_lif = MultiStepParametricLIFNode(
                init_tau=2.0, v_threshold=0.5, detach_reset=True, backend="cupy"
            )

        self.talking_heads = nn.Conv1d(
            num_heads, num_heads, kernel_size=1, stride=1, bias=False
        )
        if spike_mode == "lif":
            self.talking_heads_lif = MultiStepLIFNode(
                tau=2.0, v_threshold=0.5, detach_reset=True, backend="cupy"
            )
            # self.talking_heads_lif = MultiStepLazyStateLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend="cupy")

        elif spike_mode == "plif":
            self.talking_heads_lif = MultiStepParametricLIFNode(
                init_tau=2.0, v_threshold=0.5, detach_reset=True, backend="cupy"
            )

        self.proj_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm2d(dim)

        if spike_mode == "lif":
            self.shortcut_lif = MultiStepLIFNode(
                tau=2.0, detach_reset=True, backend="cupy"
            )
            # self.shortcut_lif = MultiStepLazyStateLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        elif spike_mode == "plif":
            self.shortcut_lif = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy"
            )

        self.mode = mode
        self.layer = layer
        # self.lif_recurrent_state = lif_recurrent_state   # changed on 2025-04-27

    def forward(self, x, hook=None):
        T, B, C, H, W = x.shape
        identity = x
        N = H * W
        # changed on 2025-04-27
        # if self.lif_recurrent_state is not None and self.lif_recurrent_state[0] == "1":
        #     tmp_x = []
        #     for t in range(T):
        #         if t == 0:
        #             x_in = x[t]    # B C H W
        #         else:
        #             x_in = x[t] + x_out
        #         x_out = self.shortcut_lif(x_in.unsqueeze(0)).squeeze(0)   # B C H W
        #         tmp_x.append(x_out)
            
        #     x = torch.stack(tmp_x, dim=0)    # T B C H W
        # else:
        x = self.shortcut_lif(x)
        
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_first_lif"] = x.detach()

        x_for_qkv = x.flatten(0, 1)
        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T, B, C, H, W).contiguous()
        # changed on 2025-04-27
        # if self.lif_recurrent_state is not None and self.lif_recurrent_state[1] == "1":
        #     tmp_x = []
        #     for t in range(T):
        #         if t == 0:
        #             x_in = q_conv_out[t]    # B C H W
        #         else:
        #             x_in = q_conv_out[t] + x_out
        #         x_out = self.q_lif(x_in.unsqueeze(0)).squeeze(0)   # B C H W
        #         tmp_x.append(x_out)
            
        #     q_conv_out = torch.stack(tmp_x, dim=0)    # T B C H W
        # else:
        q_conv_out = self.q_lif(q_conv_out)

        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_q_lif"] = q_conv_out.detach()
        q = (
            q_conv_out.flatten(3)
            .transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T, B, C, H, W).contiguous()
        # changed on 2025-04-27
        # if self.lif_recurrent_state is not None and self.lif_recurrent_state[2] == "1":
        #     tmp_x = []
        #     for t in range(T):
        #         if t == 0:
        #             x_in = k_conv_out[t]    # B C H W
        #         else:
        #             x_in = k_conv_out[t] + x_out
        #         x_out = self.k_lif(x_in.unsqueeze(0)).squeeze(0)   # B C H W
        #         tmp_x.append(x_out)
            
        #     k_conv_out = torch.stack(tmp_x, dim=0)    # T B C H W
        # else:
        k_conv_out = self.k_lif(k_conv_out)

        if self.dvs:
            k_conv_out = self.pool(k_conv_out)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_k_lif"] = k_conv_out.detach()
        k = (
            k_conv_out.flatten(3)
            .transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        v_conv_out = self.v_conv(x_for_qkv)
        v_conv_out = self.v_bn(v_conv_out).reshape(T, B, C, H, W).contiguous()
        # changed on 2025-04-27
        # if self.lif_recurrent_state is not None and self.lif_recurrent_state[3] == "1":
        #     tmp_x = []
        #     for t in range(T):
        #         if t == 0:
        #             x_in = v_conv_out[t]    # B C H W
        #         else:
        #             x_in = v_conv_out[t] + x_out
        #         x_out = self.v_lif(x_in.unsqueeze(0)).squeeze(0)   # B C H W
        #         tmp_x.append(x_out)
            
        #     v_conv_out = torch.stack(tmp_x, dim=0)    # T B C H W
        # else:
        v_conv_out = self.v_lif(v_conv_out)

        if self.dvs:
            v_conv_out = self.pool(v_conv_out)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_v_lif"] = v_conv_out.detach()
        v = (
            v_conv_out.flatten(3)
            .transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )  # T B head N C//h

        kv = k.mul(v)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_kv_before"] = kv
        if self.dvs:
            kv = self.pool(kv)
        kv = kv.sum(dim=-2, keepdim=True)           # T B head 1 C//h
        # changed on 2025-04-27
        # if self.lif_recurrent_state is not None and self.lif_recurrent_state[4] == "1":
        #     tmp_x = []
        #     for t in range(T):
        #         if t == 0:
        #             x_in = kv[t]    # B head 1 C//h
        #         else:
        #             x_in = kv[t] + x_out
        #         x_out = self.talking_heads_lif(x_in.unsqueeze(0)).squeeze(0)   # B head 1 C//h
        #         tmp_x.append(x_out)
            
        #     kv = torch.stack(tmp_x, dim=0)    # T B head 1 C//h
        # else:
        kv = self.talking_heads_lif(kv)

        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_kv"] = kv.detach()
        x = q.mul(kv)          # T B head N C//h
        if self.dvs:
            x = self.pool(x)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_x_after_qkv"] = x.detach()

        x = x.transpose(3, 4).reshape(T, B, C, H, W).contiguous()   # T B C H W
        
        x = (
            self.proj_bn(self.proj_conv(x.flatten(0, 1)))
            .reshape(T, B, C, H, W)
            .contiguous()
        )

        x = x + identity
        return x, v, hook


class MS_Block_Conv(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        sr_ratio=1,
        attn_mode="direct_xor",
        spike_mode="lif",
        dvs=False,
        layer=0,
        # lif_recurrent_state=None,     # changed on 2025-04-27
    ):
        super().__init__()
        self.attn = MS_SSA_Conv(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
            mode=attn_mode,
            spike_mode=spike_mode,
            dvs=dvs,
            layer=layer,
            # lif_recurrent_state=lif_recurrent_state[0:5],
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MS_MLP_Conv(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            drop=drop,
            spike_mode=spike_mode,
            layer=layer,
            # lif_recurrent_state=lif_recurrent_state[5:7],
        )

    def forward(self, x, hook=None):
        x_attn, attn, hook = self.attn(x, hook=hook)
        x, hook = self.mlp(x_attn, hook=hook)
        return x, attn, hook
