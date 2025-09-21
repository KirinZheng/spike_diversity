from copy import deepcopy
import numpy as np
import random
import torch
import torch.nn as nn
from spikingjelly.activation_based import functional, neuron, layer, surrogate
from spikingjelly.activation_based.model import spiking_resnet

from attack import fgsm, pgd, jitter_attack, rfgsm


def set_seed(seed):
    # 设置 Python 内置的随机数生成器的种子
    random.seed(seed)
    # 设置 NumPy 的随机数生成器的种子
    np.random.seed(seed)
    # 设置 PyTorch 随机数生成器的种子
    torch.manual_seed(seed)
    # 如果使用的是 GPU，还需要设置 GPU 上的种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果使用多个GPU
    # 设置 PyTorch 后端为确定性模式（固定运算结果）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CIFAR10Net(nn.Module):
    def __init__(self, channels=256, spiking_neuron: callable = None, **kwargs):
        super().__init__()

        conv = []
        for i in range(2):
            for j in range(3):
                if conv.__len__() == 0:
                    in_channels = 3
                else:
                    in_channels = channels

                conv.append(layer.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False))
                conv.append(layer.BatchNorm2d(channels))
                conv.append(spiking_neuron(**deepcopy(kwargs)))

            conv.append(layer.MaxPool2d(2, 2))

        self.conv_fc = nn.Sequential(
            *conv,
            layer.Flatten(),
            layer.Dropout(0.5),
            layer.Linear(channels * 8 * 8, 2048),
            spiking_neuron(**deepcopy(kwargs)),

            layer.Dropout(0.5),
            layer.Linear(2048, 100),
            spiking_neuron(**deepcopy(kwargs)),

            layer.VotingLayer(10)
        )

    def forward(self, x):
        return self.conv_fc(x).mean(1)


if __name__ == "__main__":
    set_seed(42)

    # define model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = CIFAR10Net(16, neuron.IFNode, surrogate_function=surrogate.ATan())

    net.to(device)
    functional.reset_net(net)
    functional.set_step_mode(net, 'm')

    # define datasets
    x = torch.rand([4, 2, 3, 32, 32]).to(device)
    y = torch.tensor([0, 0, 7, 2]).to(device)

    # feed-forward test
    output = net(x)
    acc = output.max(1)[1] == y
    functional.reset_net(net)

    # white-box
    # loss function define
    # loss_fn = nn.CrossEntropyLoss()
    # gen attack samples
    # adv_x = fgsm(net, x, y, loss_fn=loss_fn, eps=8 / 255)
    # adv_x = pgd(net, x, y, loss_fn=loss_fn)
    # adv_x = jitter_attack(net, x, y)
    # adv_x = rfgsm(net, x, y, loss_fn=loss_fn)

    # # 输出原始和对抗样本的对比
    # print("Original Images:", x)
    # print("Adversarial Images:", adv_x)

    # black-box
    # define a surrogate model to genarate adv-data
    net_surrogate = spiking_resnet.spiking_resnet18(pretrained=False, progress=True, spiking_neuron=neuron.IFNode)
    loss_fn = nn.CrossEntropyLoss()
    adv_x = rfgsm(net, x, y, loss_fn=loss_fn)
    # 输出原始和对抗样本的对比
    print("Original Images:", x)
    print("Adversarial Images:", adv_x)
    # adv_x are used to attack model CIFAR10Net
    # ...
