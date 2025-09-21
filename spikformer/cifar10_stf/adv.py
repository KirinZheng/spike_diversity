import torch
import torch.nn as nn
import torch.nn.functional as F
# from spikingjelly.activation_based import functional
from spikingjelly.clock_driven import functional


def fgsm(model, images, labels, loss_fn, eps=6 / 255,td = False):
    r"""
    FGSM (Fast Gradient Sign Method) 对抗攻击用于脉冲神经网络 (SNNs) 的实现。

    参数:
    - model: 被攻击的 SNN 模型
    - images: 输入图像 (batch_size, time_steps, channels, height, width)
    - labels: 正确的标签，用于损失计算
    - loss_fn: 用于计算损失的函数 (例如 CrossEntropyLoss)
    - eps: 扰动大小，控制生成对抗样本时的最大步长

    返回:
    - adv_images: 通过 FGSM 生成的对抗样本
    """

    # 将输入图像和标签复制一份并从计算图中分离，避免在攻击过程中影响原数据
    images = images.clone().detach()
    labels = labels.clone().detach()
    
    images = images.float()
    # 确保图像张量被追踪其梯度，必要时进行反向传播
    images.requires_grad = True

    # 对输入图像进行前向传播，获得模型输出
   
    if td:
        x1,td,tmp = model(images)    
        functional.reset_net(model)
        outputs, _ = model(tmp,td = td)
    else:
        outputs = model(images)

  
        # outputs = outputs.mean(0)
        

    # 计算模型输出与正确标签之间的损失
    cost = loss_fn(outputs, labels)

    # 计算损失对输入图像的梯度
    grad = torch.autograd.grad(
        cost, images, retain_graph=False, create_graph=False
    )[0]

    # 使用梯度符号法则更新对抗样本 (FGSM 核心公式: 原始图像 + eps * 梯度符号)
    adv_images = images + eps * grad.sign()

    # 将生成的对抗样本的像素值裁剪到合法范围内 (通常在 0 和 1 之间)
    adv_images = torch.clamp(adv_images, min=0, max=1).detach()

    # 重置 SNN 的状态，确保模型的状态在下一个输入前被清除
    functional.reset_net(model)

    # 返回生成的对抗样本
    return adv_images


def pgd(model, images, labels, loss_fn, eps=8 / 255, alpha=2 / 255, iters=10,td = False):
    r"""
    Projected Gradient Descent (PGD) Attack for SNNs.

    Arguments:
    - model: the SNN model
    - images: the input images
    - labels: the true labels
    - loss_fn: loss function used to calculate the adversarial loss
    - eps: maximum perturbation (L∞ norm)
    - alpha: step size for each iteration
    - iters: number of iterations

    Returns:
    - adv_images: adversarial images after PGD attack
    """

    # Clone and detach to ensure gradients aren't tracked unnecessarily
    images = images.clone().detach()
    labels = labels.clone().detach()

    # Initialize adversarial images as the original input
    adv_images = images.clone().detach()

    # Add a small random perturbation to the initial adversarial images
    adv_images = adv_images + torch.empty_like(adv_images).uniform_(-eps, eps)
    adv_images = torch.clamp(adv_images, min=0, max=1).detach()  # Ensure the values are within valid range

    for _ in range(iters):
        adv_images.requires_grad = True  # Ensure the adversarial images require gradients

        # Get model outputs for the adversarial images
        # outputs = model(adv_images)

    

        if td:
            x1,td,tmp = model(adv_images)    
            functional.reset_net(model)
            outputs, _  = model(tmp,td = td)
        else:
            outputs = model(adv_images)

        # outputs = outputs.mean(0)
      

        cost = loss_fn(outputs, labels)


        # Compute the gradients of the loss with respect to the adversarial images
        grad = torch.autograd.grad(cost, adv_images, retain_graph=False, 
                                   create_graph=False)[0]

        # Update adversarial images with the gradient sign method and step size alpha
        adv_images = adv_images + alpha * grad.sign()

        # Clamp the adversarial images to ensure they're within the epsilon ball around the original images
        delta = torch.clamp(adv_images - images, min=-eps, max=eps)
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()  # detach here is fine

        # Reset the SNN's state at each iteration
        functional.reset_net(model)

    return adv_images


def jitter_attack(model, images, labels, eps=8 / 255, alpha=2 / 255, iters=10, scale=10, std=0.1, random_start=True):
    r"""
    Jitter Attack function for adversarial attacks on spiking data.

    Arguments:
    - model (nn.Module): model to attack.
    - images (torch.Tensor): input spike data of shape (batch_size, T, channel, height, width).
    - labels (torch.Tensor): true labels of the input data.
    - eps (float): maximum perturbation (Linf norm).
    - alpha (float): step size for each iteration.
    - steps (int): number of iterations.
    - scale (float): scaling factor for logits.
    - std (float): standard deviation for adding noise to logits.
    - random_start (bool): whether to start with a random perturbation.

    Returns:
    - adv_images (torch.Tensor): adversarial spike data after applying the jitter attack.
    """
    device = images.device
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)

    loss_fn = nn.MSELoss(reduction="none")
    adv_images = images.clone().detach()

    if random_start:
        # Starting at a uniformly random point
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(-eps, eps)
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

    for _ in range(iters):
        adv_images.requires_grad = True


        # logits = model(adv_images)
        if td:

            x1,td,tmp = model(images)    
            functional.reset_net(model)
            logits,rencon_loss = model(tmp,td = td)
        
        else:
            logits = model(images)

            
        _, predictions = torch.max(logits, dim=1)
        wrong = predictions != labels

        norm_z = torch.norm(logits, p=float("inf"), dim=1, keepdim=True)
        hat_z = F.softmax(scale * logits / norm_z, dim=1)

        if std != 0:
            hat_z = hat_z + std * torch.randn_like(hat_z)

        Y = F.one_hot(labels, num_classes=logits.shape[-1]).float()
        cost = loss_fn(hat_z, Y).mean(dim=1)

        norm_r = torch.norm((adv_images - images), p=float("inf"), dim=[1, 2, 3, 4])
        nonzero_r = norm_r != 0
        cost[wrong & nonzero_r] /= norm_r[wrong & nonzero_r]

        cost = cost.mean()

        # Update adversarial spike data
        grad = torch.autograd.grad(cost, adv_images, retain_graph=False, create_graph=False)[0]
        adv_images = adv_images.detach() + alpha * grad.sign()
        delta = torch.clamp(adv_images - images, min=-eps, max=eps)
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()
        # Reset the SNN's state at each iteration
        functional.reset_net(model)

    return adv_images