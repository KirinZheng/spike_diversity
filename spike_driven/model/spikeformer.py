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
from module import *


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
        diversity_loss=False,           # changed on 2025-04-23
        lif_recurrent_state=None,       # changed on 2025-04-27
    ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        self.T = T
        self.TET = TET
        self.dvs = dvs_mode

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depths)
        ]  # stochastic depth decay rule
        self.recurrent_coding = recurrent_coding    # changed on 2025-04-13
        self.pe_type=pe_type                        # changed on 2025-04-17
        self.diversity_loss = diversity_loss        # changed on 2025-04-23
        self.lif_recurrent_state = lif_recurrent_state  # changed on 2025-04-27
        
        
        if self.lif_recurrent_state is not None:
            lif_recurrent_state_length = 4 + depths * 7
            assert len(self.lif_recurrent_state) == lif_recurrent_state_length, f"Sorry the length of lif_recurrent_state must be \
                {lif_recurrent_state_length}, while the current length is {len(self.lif_recurrent_state)}!"

            lif_recurrent_state_index = 0

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
            use_diversity_loss=diversity_loss,
            lif_recurrent_state = lif_recurrent_state[0:3]
        )

        if self.lif_recurrent_state is not None:
            print(f"MS_SPS: LIF index: {', '.join(map(str, range(0, 3)))}")
            print(f"MS_SPS: LIF using recurrent is: {lif_recurrent_state[0:3]}")

            lif_recurrent_state_index = lif_recurrent_state_index + 3


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
                    lif_recurrent_state=lif_recurrent_state[(lif_recurrent_state_index+(j)*7):(lif_recurrent_state_index+(j+1)*7)]
                )
                for j in range(depths)
            ]
        )

        if self.lif_recurrent_state is not None:
            for i in range(depths):
                print(f"MS_Block_Conv Block{i}: LIF index: {', '.join(map(str, range(lif_recurrent_state_index+(i)*7, lif_recurrent_state_index+(i+1)*7)))}")
                print(f"MS_Block_Conv Block{i}: LIF using recurrent is: {lif_recurrent_state[(lif_recurrent_state_index+(i)*7):(lif_recurrent_state_index+(i+1)*7)]}")


            lif_recurrent_state_index = lif_recurrent_state_index + (depths) * 7

            self.head_lif_recurrent_state_index = lif_recurrent_state_index

        if self.lif_recurrent_state is not None:
            print(f"Head LIF: LIF index: {self.head_lif_recurrent_state_index}")
            print(f"Head LIF: LIF using recurrent is: {lif_recurrent_state[self.head_lif_recurrent_state_index]}")

        setattr(self, f"patch_embed", patch_embed)
        setattr(self, f"block", blocks)

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

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x, hook=None):
        block = getattr(self, f"block")
        patch_embed = getattr(self, f"patch_embed")
        # changed on 2025-0-24
        if self.diversity_loss:
            x, _, hook, diversity_coding = patch_embed(x, hook=hook)
        else:
            x, _, hook = patch_embed(x, hook=hook)
        for blk in block:
            x, _, hook = blk(x, hook=hook)

        x = x.flatten(3).mean(3)

        if self.diversity_loss:
            return x, hook, diversity_coding
        else:
            return x, hook

    def forward(self, x, hook=None):
        if len(x.shape) < 5:
            x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
        else:
            x = x.transpose(0, 1).contiguous()
        # changed on 2025-0-24
        if self.diversity_loss:
            x, hook, diversity_coding = self.forward_features(x, hook=hook)
        else:
            x, hook = self.forward_features(x, hook=hook)
        # changed on 2025-04-27
        if self.lif_recurrent_state is not None and self.lif_recurrent_state[self.head_lif_recurrent_state_index] == "1":
            tmp_x = []
            for t in range(self.T):
                if t == 0:
                    x_in = x[t]    # B C H W
                else:
                    x_in = x[t] + x_out
                x_out = self.head_lif(x_in.unsqueeze(0)).squeeze(0)   # B C H W
                tmp_x.append(x_out)
            
            x = torch.stack(tmp_x, dim=0)    # T B C H W
        else:
            x = self.head_lif(x)
        if hook is not None:
            hook["head_lif"] = x.detach()

        x = self.head(x)
        if not self.TET:
            x = x.mean(0)
            
        if self.diversity_loss:
            return x, hook, diversity_coding
        else:
            return x, hook


@register_model
def sdt(**kwargs):
    model = SpikeDrivenTransformer(
        **kwargs,
    )
    model.default_cfg = _cfg()
    print(f"using recurrent coding: {model.recurrent_coding}")
    print(f"postion embedding methods: {model.pe_type}")
    return model
