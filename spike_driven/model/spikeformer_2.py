from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from spikingjelly.clock_driven.neuron import (
    MultiStepLIFNode,
    MultiStepParametricLIFNode,
)
from torch import Tensor
from einops import rearrange
from module import *


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
    def __init__(self, lidx: int, dim: int, C: int, last_layer: bool=False, expand_last: bool=False, round64: bool=True) -> None:
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


class SpikeDrivenTransformer(nn.Module):
    def __init__(
        self,
        img_size_h=128,
        img_size_w=128,
        patch_size=16,
        in_channels=2,
        num_classes=11,
        embed_dims=512,
        num_heads=8,
        mlp_ratios=4,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        depths=[6, 8, 6],
        sr_ratios=[8, 4, 2],
        T=4,
        pooling_stat="1111",
        attn_mode="direct_xor",
        spike_mode="lif",
        get_embed=False,
        dvs_mode=False,
        TET=False,
        cml=False,
        pretrained=False,
        pretrained_cfg=None,
        recurrent_coding=False,         # changed on 2025-04-13
        recurrent_lif=None,             # changed on 2025-04-13
        pe_type=None,                   # changed on 2025-04-17
        # diversity_loss=False,           # changed on 2025-04-23
        # lif_recurrent_state=None,       # changed on 2025-04-27
        temporal_conv_type=None,        # changed on 2025-09-22
        maxpooling_lif_change_order=False,
        dense_connection=False,
        dense_easy_connection=False,
        dense_dynamic_fixed=False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        self.T = T
        self.TET = TET
        self.dvs = dvs_mode

        assert pe_type in ("stf_1", "stf_2"), f"Invalid pe_type: {pe_type}, must be 'stf_1' or 'stf_2'"
        assert temporal_conv_type in ("conv1d", "conv2d"), f"Invalid temporal_conv_type: {temporal_conv_type}, must be 'conv1d' or 'conv2d'"
        assert (not dense_connection and not dense_easy_connection) or (dense_connection ^ dense_easy_connection), \
                "Invalid config: set at most one of 'dense_connection' and 'dense_easy_connection' to True."

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depths)
        ]  # stochastic depth decay rule
        self.recurrent_coding = recurrent_coding    # changed on 2025-04-13
        self.recurrent_lif = recurrent_lif
        self.temporal_conv_type=temporal_conv_type
        self.pe_type=pe_type                        # changed on 2025-04-17
        self.maxpooling_lif_change_order = maxpooling_lif_change_order
        self.dense_easy_connection=dense_easy_connection
        self.dense_connection = dense_connection
        self.dense_dynamic_fixed=dense_dynamic_fixed

        # self.diversity_loss = diversity_loss        # changed on 2025-04-23
        # self.lif_recurrent_state = lif_recurrent_state  # changed on 2025-04-27
        
        
        # if self.lif_recurrent_state is not None:
        #     lif_recurrent_state_length = 4 + depths * 7
        #     assert len(self.lif_recurrent_state) == lif_recurrent_state_length, f"Sorry the length of lif_recurrent_state must be \
        #         {lif_recurrent_state_length}, while the current length is {len(self.lif_recurrent_state)}!"

        #     lif_recurrent_state_index = 0
        if not maxpooling_lif_change_order:
            patch_embed = MS_SPS(
                img_size_h=img_size_h,
                img_size_w=img_size_w,
                patch_size=patch_size,
                in_channels=in_channels,
                embed_dims=embed_dims,
                pooling_stat=pooling_stat,
                spike_mode=spike_mode,
                recurrent_coding=recurrent_coding,
                recurrent_lif=recurrent_lif,
                pe_type=pe_type,
                time_step=T,
                temporal_conv_type=temporal_conv_type
                # use_diversity_loss=diversity_loss,
                # lif_recurrent_state = lif_recurrent_state[0:3]
            )
        elif maxpooling_lif_change_order:
            patch_embed = MS_SPS_Maxpooling_LIF_changed(
                img_size_h=img_size_h,
                img_size_w=img_size_w,
                patch_size=patch_size,
                in_channels=in_channels,
                embed_dims=embed_dims,
                pooling_stat=pooling_stat,
                spike_mode=spike_mode,
                recurrent_coding=recurrent_coding,
                recurrent_lif=recurrent_lif,
                pe_type=pe_type,
                time_step=T,
                temporal_conv_type=temporal_conv_type
                # use_diversity_loss=diversity_loss,
                # lif_recurrent_state = lif_recurrent_state[0:3]
            )
        else:
            assert False, "Sorry, that's wrong!"

        # if self.lif_recurrent_state is not None:
        #     print(f"MS_SPS: LIF index: {', '.join(map(str, range(0, 3)))}")
        #     print(f"MS_SPS: LIF using recurrent is: {lif_recurrent_state[0:3]}")

        #     lif_recurrent_state_index = lif_recurrent_state_index + 3


        blocks = nn.ModuleList(
            [
                MS_Block_Conv(
                    dim=embed_dims,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratios,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[j],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios,
                    attn_mode=attn_mode,
                    spike_mode=spike_mode,
                    dvs=dvs_mode,
                    layer=j,
                    # lif_recurrent_state=lif_recurrent_state[(lif_recurrent_state_index+(j)*7):(lif_recurrent_state_index+(j+1)*7)]
                )
                for j in range(depths)
            ]
        )

        if self.dense_connection:
            # MS_Block_Conv qkvr C=4
            blocks_da = nn.ModuleList([DynamiceResidualBlock(lidx=lidx, dim=embed_dims, C=4, 
                last_layer=lidx==depths-1, expand_last=True, round64=True) for lidx in range(depths)])
            blocks_bs = nn.ParameterList([nn.Parameter(data=torch.randn(4 if lidx != depths-1 else 1, lidx+2)) for lidx in range(depths)])

        if self.dense_easy_connection:
            self.n_repeat = (depths) // 1
            self.dilation_factor = 1
            self.increate_T_every = 1
            self.weights = nn.ModuleList([
                    nn.Linear((i + 2 + self.dilation_factor - 1) // self.dilation_factor, 1, bias=False) 
                    for i in range(self.n_repeat)
                ])
            
            if self.dense_dynamic_fixed:
                for m in self.weights:
                    m.weight.requires_grad = False


        # if self.lif_recurrent_state is not None:
        #     for i in range(depths):
        #         print(f"MS_Block_Conv Block{i}: LIF index: {', '.join(map(str, range(lif_recurrent_state_index+(i)*7, lif_recurrent_state_index+(i+1)*7)))}")
        #         print(f"MS_Block_Conv Block{i}: LIF using recurrent is: {lif_recurrent_state[(lif_recurrent_state_index+(i)*7):(lif_recurrent_state_index+(i+1)*7)]}")


        #     lif_recurrent_state_index = lif_recurrent_state_index + (depths) * 7

        #     self.head_lif_recurrent_state_index = lif_recurrent_state_index

        # if self.lif_recurrent_state is not None:
        #     print(f"Head LIF: LIF index: {self.head_lif_recurrent_state_index}")
        #     print(f"Head LIF: LIF using recurrent is: {lif_recurrent_state[self.head_lif_recurrent_state_index]}")

        setattr(self, f"patch_embed", patch_embed)
        setattr(self, f"block", blocks)
        if self.dense_connection:
            setattr(self, f"block_da", blocks_da)
            setattr(self, f"block_bs", blocks_bs)

        # classification head
        if spike_mode in ["lif", "alif", "blif"]:
            self.head_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        elif spike_mode == "plif":
            self.head_lif = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy"
            )
        self.head = (
            nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        )
        self.apply(self._init_weights)

        if self.dense_easy_connection:
            for module in self.weights:
                module.weight.data.zero_()
                module.weight.data[:, :] = 1.       # all dynamic  factor to be 1.0

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x, use_dense_connection=False, hook=None):
        block = getattr(self, f"block")
        patch_embed = getattr(self, f"patch_embed")
        if self.dense_connection and use_dense_connection:
            block_da = getattr(self, f"block_da")
            block_bs = getattr(self, f"block_bs")

        # changed on 2025-0-24
        # if self.diversity_loss:
        #     x, _, hook, diversity_coding = patch_embed(x, hook=hook)
        # else:
        x, _, hook = patch_embed(x, hook=hook)

        if self.dense_connection:

            if use_dense_connection:
                hiddens = [x] # [T B C H W]
                idx = 0

            for blk in block:
                x, _, hook = blk(x, hook=hook)
                if use_dense_connection:
                    hiddens.append(x)       # T B C H W
                    dw = block_da[idx](x)   # 4 T B lidx+2 H W
                    dw = dw + block_bs[idx][:, None, None, :, None, None]  # 4 T B lidx+2 H W
                    x = block_da[idx].layer_mix(hiddens, dw)
                    idx += 1

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

                x_v = None
                for rep_idx in range(1, self.n_repeat + 1):
                    for i in range(self.increate_T_every):
                        x, _, hook = block[(rep_idx - 1) * self.increate_T_every  + i](x, x_v=x_v)
                    hiddens[rep_idx % self.dilation_factor] = apply_inplace_set(
                        hiddens[rep_idx % self.dilation_factor], 
                        rep_idx // self.dilation_factor, 
                        x,
                    )
                    x_v = torch.tensordot(self.weights[rep_idx - 1].weight.view(-1), 
                                        hiddens[rep_idx % self.dilation_factor][1], dims=1)

        else:
            for blk in block:
                x, _, hook = blk(x, hook=hook)
        
        x = x.flatten(3).mean(3)
        # if self.diversity_loss:
        #     return x, hook, diversity_coding
        # else:
        return x, hook

    def forward(self, x, use_dense_connection=False, hook=None):
        if len(x.shape) < 5:
            x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
        else:
            x = x.transpose(0, 1).contiguous()
        # changed on 2025-0-24
        # if self.diversity_loss:
        #     x, hook, diversity_coding = self.forward_features(x, hook=hook)
        # else:
        x, hook = self.forward_features(x, use_dense_connection=use_dense_connection, hook=hook)
        # changed on 2025-04-27
        # if self.lif_recurrent_state is not None and self.lif_recurrent_state[self.head_lif_recurrent_state_index] == "1":
        #     tmp_x = []
        #     for t in range(self.T):
        #         if t == 0:
        #             x_in = x[t]    # B C H W
        #         else:
        #             x_in = x[t] + x_out
        #         x_out = self.head_lif(x_in.unsqueeze(0)).squeeze(0)   # B C H W
        #         tmp_x.append(x_out)
            
        #     x = torch.stack(tmp_x, dim=0)    # T B C H W
        # else:
        x = self.head_lif(x)
        if hook is not None:
            hook["head_lif"] = x.detach()

        x = self.head(x)
        if not self.TET:
            x = x.mean(0)
            
        # if self.diversity_loss:
        #     return x, hook, diversity_coding
        # else:
        return x, hook


@register_model
def sdt_2(**kwargs):
    model = SpikeDrivenTransformer(
        **kwargs,
    )
    model.default_cfg = _cfg()
    print("dense finegrained")
    print(f"dense_dynamic_fixed: {model.dense_dynamic_fixed}")
    print(f"using recurrent coding: {model.recurrent_coding}")
    print(f"pe_type: {model.pe_type}")
    print(f"temporal_conv_type: {model.temporal_conv_type}")
    print(f"maxpooling_lif_change_order: {model.maxpooling_lif_change_order}")
    print(f"dense_connection: {model.dense_connection}")
    print(f"dense_easy_connection: {model.dense_easy_connection}")
    return model
