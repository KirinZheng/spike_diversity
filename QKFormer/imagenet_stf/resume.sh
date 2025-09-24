## official weight 10-768 T=4 224
# /zhengzeqi/code/zhuzizheng/122Qkformer/official_weight/T4_224offical.pth
## official weight 10-768 T=4 288
# /zhengzeqi/code/zhuzizheng/122Qkformer/official_weight/T4_288official.pth
## official weight 10-768 T=4 384
# /zhengzeqi/code/zhuzizheng/122Qkformer/official_weight/T4_384official.pth



## official weight 10-384 T=4 224
# /zhengzeqi/code/zhuzizheng/122Qkformer/official_weight/10_384_T_4_224.pth
## official weight 10-512 T=4 224
# /zhengzeqi/code/zhuzizheng/122Qkformer/official_weight/10_512_T_4_224.pth

# 设置显存空闲阈值（单位：MiB）
THRESHOLD=90000
NUM_GPUS=8

# 检查函数
check_gpu_free_mem() {
    echo "🔍 正在检查 GPU 显存是否每张卡空闲超过 ${THRESHOLD} MiB..."
    FREE_MEMS=($(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits))
    
    for ((i=0; i<$NUM_GPUS; i++)); do
        FREE=${FREE_MEMS[$i]}
        echo "GPU $i 空闲显存：${FREE} MiB"
        if [ "$FREE" -lt "$THRESHOLD" ]; then
            echo "❌ GPU $i 显存不足，等待中..."
            return 1
        fi
    done

    return 0
}

# 等待直到显存足够
while true; do
    if check_gpu_free_mem; then
        echo "所有 GPU 显存充足，开始运行训练任务..."
        break
    else
        echo "等待 120 秒后重试..."
        sleep 120
    fi
done


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port 29502 /zhengzeqi/code/zhengzeqi/QKFormer_original/imagenet_1K_finetune/train.py \
    --output_dir /zhengzeqi/code/zhengzeqi/QKFormer_original/imagenet_1K_finetune/output/10_768_T_4_384_3d_pe_arch_4_resume \
    --log_dir /zhengzeqi/code/zhengzeqi/QKFormer_original/imagenet_1K_finetune/output/10_768_T_4_384_3d_pe_arch_4_resume \
    --data_path /zhengzeqi/dataset/imagenet \
    --model QKFormer_10_768 \
    --input_size 384 \
    --time_step 4 \
    --batch_size 22 \
    --accum_iter 3 \
    --epochs 10 \
    --lr 2e-05 \
    --warmup_epochs 0 \
    --recurrent_coding \
    --pe_type 3d_pe_arch_4 \
    --resume /zhengzeqi/code/zhengzeqi/QKFormer_original/imagenet_1K_finetune/output/10_768_T_4_224_3d_pe_arch_4_finetune/checkpoint-23.pth