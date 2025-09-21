#!/bin/bash
function cleanup {
    echo "Received termination signal. Cleaning up..."
    # 可以在这里添加需要的清理操作，如终止正在运行的Python进程等
    # 例如，可以尝试终止所有后台Python进程
    pkill -P $$  # 终止与当前进程相关的子进程
    exit 1
}

# 捕获SIGINT信号（Ctrl+C）并调用cleanup函数
trap cleanup SIGINT
# 运行任务
CUDA_VISIBLE_DEVICES=0 python train_d.py -T 2 --seed 42 --output ./output/cifar100/T2/op2/3407 --experiment 3407 --op 2 -c ./cifar100.yml  &
# CUDA_VISIBLE_DEVICES=1 python train_d.py --seed 42 --experiment CM4V1 --qkv 3 --V 1 &
# CUDA_VISIBLE_DEVICES=2 python train_d.py --seed 42 --output ./718 --experiment CM4V3 --qkv 3 --V 3 &
# CUDA_VISIBLE_DEVICES=3 python train_d.py --seed 42 --output ./718 --experiment CM4V4 --qkv 3 --V 4 &

# 等待所有后台任务完成
wait
