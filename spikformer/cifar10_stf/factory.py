from torch import Tensor
import torch
import torch.nn as nn
from torch.nn.common_types import _size_2_t
from torch.nn.modules.utils import _pair
from torch.nn import functional as F
from typing import List
from spikingjelly.clock_driven.neuron import MultiStepLIFNode,MultiStepParametricLIFNode
from functools import partial
from timm.models.layers import trunc_normal_   #timm==0.5.4







class conv2d_td(nn.modules.conv._ConvNd):
    __doc__ = r"""Applies a 2D convolution over an input signal composed of serveral input plates.
    Comparing to nn.Conv2d, con2d_td has one more top-down tensor as input.
    Other paramaters remain same with nn.Conv2d.


    top-down:
    the top-down tensor should be (out_channels,in_channels,kernel_size,kernel_size)

    operator:
    the top-down tensor(td) will add to the kernal weight or will mutiplicate with kernal weight depend on parameter operator("str")
    operator can be ['nothing','mul','add','cos_mask'], and when operator is 'mask',the weight and the td will calulate the cos similiarity and
    then throught MLP layer to get the tensor which multiplicate with the weight 

    target:
    td can operate with weight tensor or input tensor or both
    so the parameter target decide which tensor will be modify by top-down tensor
    target is a List[str], it can be ['weight','input','both']
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = False,
        padding_mode: str = 'zeros', 
        device=None,
        dtype=None,
        operator = 'nothing',
        target = 'weight',
        aug = 1
    
    ) -> None:
        kwargs = {'device':device,'dtype':dtype}
        tar_list = ['weight','input']
        oper_list = ['nothing','mul','add','cos_mask']                                      
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = _pair(padding)
        dilation_ = _pair(dilation)
        #调用cov2d_td的构造函数
        super(conv2d_td, self).__init__(
             in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _pair(0),groups, bias, padding_mode, **kwargs)
        self.operator = operator
        self.target = target
        self.aug = aug

        assert self.operator in oper_list,  f"invalid input of operator, must be in {oper_list}"
        assert self.target in tar_list,  f"invalid input of target, must be in {tar_list}"
        
    

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Tensor,td:Tensor) -> Tensor:
        if td is None:
            return F.conv2d(input, weight, bias, self.stride,self.padding,self.dilation,self.groups)
        # if self.padding_mode != 'zeros':
        #     return F.conv2d(F.pad(input, self._padding_repeated_twice, mode=self.padding_mode),
        #                     weight, bias, self.stride,
        #                     _pair(0), self.dilation, self.groups)
        if self.operator == 'nothing' or td is None:
            return F.conv2d(input, weight, bias, self.stride,
                            self.padding, self.dilation, self.groups)
        if self.operator == 'mul':
            if self.target == 'weight':
                return F.conv2d(input, weight * td, bias, self.stride,
                            self.padding, self.dilation, self.groups)
            if self.target == 'input':
                return F.conv2d(input * td, weight, bias, self.stride,
                            self.padding, self.dilation, self.groups)
        if self.operator == 'add':
            if self.target == 'weight':
                return F.conv2d(input, weight + td, bias, self.stride,
                            self.padding, self.dilation, self.groups)
            if self.target == 'input':
                return F.conv2d(input + td, weight, bias, self.stride,
                            self.padding, self.dilation, self.groups)
        if self.operator == 'cos_mask': 
            cos_sim = self.weight @ td 
            mask = cos_sim.clamp(0,1)
            if self.target == 'weight':
                return F.conv2d(input, weight * mask * self.aug, bias, self.stride,
                            self.padding, self.dilation, self.groups)
            if self.target == 'input':
                return F.conv2d(input * mask * self.aug, weight, bias, self.stride,
                            self.padding, self.dilation, self.groups)
    
    def forward(self,input:Tensor,td= None) -> Tensor:
        return self._conv_forward(input, self.weight, self.bias,td)




class Attention_td(nn.Module):
    def __init__(self, dim, num_heads=8, bias=False, att_drop=False, proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.attn_drop = nn.Dropout(att_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self,x,td = None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if td is not None:
            qkv_td = self.qkv(td).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            v = v + qkv_td[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

        
class Betascheduler:
    def __init__(self,initial,final,total_epochs):
        self.initial = initial
        self.final = final
        self.total_epochs = total_epochs
        self.currrent_epochs = 0
        self.beta = initial

    def step(self):

        self.beta += (self.final - self.initial)*(1/self.total_epochs)
        self.currrent_epochs += 1

    def get(self):
        return self.beta




class Decoder_MLP(nn.Module):
    def __init__(self,in_channels,hidden_channels,T,lif_mode ='lif',norm_layer_td = 'batch'):
        super().__init__()

        # if norm_layer_td == 'batch':
        #     print("using batch")
        #     self.bn_td = nn.BatchNorm1d(hiddenchannels)
        # elif norm_layer_td == 'layer':
        #     print("using layer")
        #     norm_layer_td = partial(nn.LayerNorm,eps=1e-06)
        #     self.bn_td = norm_layer_td(out_channels)
        # elif norm_layer_td == 'none':
        #     print("do not use norm layer")
        #     self.bn_td = nn.Identity()
        # else:
        #     print("no available norm_layer")
        #     exit(0)

        self.linear_1 = nn.Linear(in_channels,hidden_channels,bias=False)
        self.bn_1 = nn.BatchNorm1d(hidden_channels)
        self.lif_1 =  MultiStepLIFNode(tau = 2.0,detach_reset = True,backend = 'torch')
        self.T = T

        self.linear_2 = nn.Linear(hidden_channels,in_channels,bias=False)
        self.bn_2 = nn.BatchNorm1d(in_channels)
        self.lif_2 =  MultiStepLIFNode(tau = 2.0,detach_reset = True,backend = 'torch')
        self.hidden = hidden_channels

        # if lif_mode == 'lif':
        #     print("using lif")
        #     self.lif_td = MultiStepLIFNode(tau = 2.0,detach_reset = True,backend = 'torch')
        # elif lif_mode == 'plif':
        #     print("using plif")
        #     self.lif_td = MultiStepParametricLIFNode(init_tau=2.0,detach_reset=True,backend='torch')
        # else:
        #      print("no available lif mode")
    def forward(self,x):   
        _ ,N,C = x.shape
     
        #Bn层接受B C 和 B C L的输入
        x = self.linear_1(x).transpose(-1,-2).contiguous()  # TB 4C N

        x = self.bn_1(x).reshape(self.T,-1,self.hidden,N).transpose(-1,-2).contiguous()  # T B 4C N

        x = self.lif_1(x).reshape(-1,self.hidden,N).transpose(-1,-2).contiguous()  # TB N 4C

        x = self.linear_2(x).transpose(-1,-2).contiguous() # TB C N

        x = self.bn_2(x).reshape(self.T,-1,C,N).transpose(-1,-2).contiguous()  # T B C N

        x = self.lif_2(x).reshape(-1,C,N).transpose(-1,-2).contiguous()  # TB N C
       
        return x














class Decoder1(nn.Module):
    def __init__(self,in_channels,out_channels,T,lif_mode ='lif',norm_layer_td = 'batch'):
        super().__init__()

        if norm_layer_td == 'batch':
            print("using batch")
            self.bn_td = nn.BatchNorm1d(out_channels)
            # self.bn = nn.BatchNorm1d(out_channels)

        elif norm_layer_td == 'layer':
            print("using layer")
            norm_layer_td = partial(nn.LayerNorm,eps=1e-06)
            self.bn_td = norm_layer_td(out_channels)

        elif norm_layer_td == 'none':
            print("do not use norm layer")
            self.bn_td = nn.Identity()

        else:
            print("no available norm_layer")
            exit(0)

        self.out_channels = out_channels
        self.linear_td = nn.Linear(in_channels,out_channels,bias=False)
        # self.linear = nn.Linear(in_channels,out_channels,bias=False)
        # self.linear_td_2 = nn.Linear(in_channels,out_channels,bias=False)
 
        self.T = T

        if lif_mode == 'lif':
            print("using lif")
            self.lif_td = MultiStepLIFNode(tau = 2.0,detach_reset = True,backend = 'torch')
        elif lif_mode == 'plif':
            print("using plif")
            self.lif_td = MultiStepParametricLIFNode(init_tau=2.0,detach_reset=True,backend='torch')
        else:
            print("no available lif mode")

        # self.apply(self._init_weights)



    # def _init_weights(self, m):
    #     print(m)
    #     if isinstance(m, nn.Linear):
    #         trunc_normal_(m.weight, std=.02)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)
        # elif hasattr(m, 'init_weights'):
        #     m.init_weights()

        



    def forward(self,x):   
        _ ,N,C = x.shape
        #接受B C 和 B C L的输入
        x = self.linear_td(x).transpose(-1,-2).contiguous()  # TB C N

        # x = self.linear(x).transpose(-1,-2).contiguous()  # TB C N

        x = self.bn_td(x).reshape(self.T,-1,C,N).transpose(-1,-2).contiguous()  # T B C N

        # x = self.bn(x).reshape(self.T,-1,C,N).transpose(-1,-2).contiguous()  # T B C N

        x = self.lif_td(x).reshape(-1,C,N).transpose(-1,-2).contiguous()  # TB N C
        return x
    


class Decoder2(nn.Module):
    def __init__(self,in_channels,out_channels,T,lif_mode ='lif',norm_layer_td = 'batch'):
        super().__init__()

        if norm_layer_td == 'batch':
            print("using batch")
            self.bn_td = nn.BatchNorm1d(out_channels)
        elif norm_layer_td == 'layer':
            print("using layer")
            norm_layer_td = partial(nn.LayerNorm,eps=1e-06)
            self.bn_td = norm_layer_td(out_channels)
        elif norm_layer_td == 'none':
            print("do not use norm layer")
            self.bn_td = nn.Identity()
        else:
            print("no available norm_layer")
            exit(0)

        self.out_channels = out_channels
        self.linear_td = nn.Linear(in_channels,out_channels,bias=False)
       
 
        self.T = T
        print("using Decoder2")
        if lif_mode == 'lif':
            print("using lif")
            self.lif_td = MultiStepLIFNode(tau = 2.0,detach_reset = True,backend = 'torch')
        elif lif_mode == 'plif':
            print("using plif")
            self.lif_td = MultiStepParametricLIFNode(init_tau=2.0,detach_reset=True,backend='torch')
        else:
            print("no available lif mode")
            self.lif_td = nn.Identity()
        # print(Decoder2)

    def forward(self,x):   
        TB,N,C = x.shape
   
        # x = x.flatten(0,1).flatten(2,3).transpose(-1,-2)   #TB N C

        #接受B C 和 B C L的输入
        # x = self.bn_td(x.transpose(-1,-2).contiguous()) # TB C N
        x = self.bn(x.transpose(-1,-2).contiguous()) # TB C N
       

        # x = self.linear_td(x.transpose(-1,-2)).reshape(self.T,-1,N,C).contiguous()  # T B N C
        x = self.linear(x.transpose(-1,-2)).reshape(self.T,-1,N,C).contiguous()  # T B N C

       
        x = self.lif_td(x).contiguous()  # T B N C

        x = x.flatten(0,1)   # TB N C

        return x
    



class Decoder_sdt(nn.Module):
    def __init__(self,in_channels,out_channels,T,lif_mode ='lif',norm_layer_td = 'batch'):
        super().__init__()

        self.conv_td = nn.Conv2d(in_channels,out_channels, kernel_size=1, stride=1)

        self.bn_td = nn.BatchNorm2d(out_channels)

        self.out_channels = out_channels
        self.T = T


        print("using Decoder_sdt")
        if lif_mode == 'lif':
            print("using lif")
            self.lif_td = MultiStepLIFNode(tau = 2.0,detach_reset = True,backend = 'torch')
        elif lif_mode == 'plif':
            print("using plif")
            self.lif_td = MultiStepParametricLIFNode(init_tau=2.0,detach_reset=True,backend='torch')
        else:
            print("no available lif mode")
            self.lif_td = nn.Identity()
        # print(Decoder_sdt)

    def forward(self,x):   
        TB,N,C = x.shape
        H,W = 8,8
        x = x.reshape(self.T,TB//self.T,N,C).transpose(-1,-2)   #T B C N
        x = x.reshape(self.T,TB//self.T,C,H,W)          # T B C H W
        
        #接受B C 和 B C L的输入
        x = self.lif_td(x)

        x = self.conv_td(x.flatten(0, 1))       # TB C H W

        x = self.bn_td(x).reshape(TB,C,N).transpose(-1,-2).contiguous()   # TB N C

        return x


