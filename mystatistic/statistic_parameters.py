import torch

def count_params_from_ckpt(ckpt_path):
    # 加载 checkpoint
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    
    # 取出 state_dict
    state_dict = checkpoint['state_dict']
    
    # 逐层统计
    total_params = 0
    for name, param in state_dict.items():
        num_params = param.numel()
        total_params += num_params
        print(f"{name:50s} {tuple(param.shape)} -> {num_params}")
    
    print("=" * 80)
    print(f"Total parameters: {total_params:,}")

# 使用方法
count_params_from_ckpt("/zhengzeqi/spike_diversity/spikformer/cifar10_stf/outputs/cifar10/T_4/stf_1_conv1d/model_best.pth.tar")