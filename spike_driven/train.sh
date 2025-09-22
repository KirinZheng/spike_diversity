# python -m torch.distributed.launch --nproc_per_node=2 /zhengzeqi/top_down/spikedriven/github_version/train.py \
# -c /zhengzeqi/top_down/spikedriven/github_version/conf/tiny_imagenet/6_512_300E_t4.yml \
# --model sdt \
# --output /zhengzeqi/top_down/spikedriven/github_version/weights \
# --time-steps 4 \
# --batch-size 128 \
# --val-batch-size 128 \
# --experiment tiny_imagenet_T_4_baseline

# python -m torch.distributed.launch --nproc_per_node=4 /zhengzeqi/top_down/spikedriven/github_version/train.py \
# -c /zhengzeqi/top_down/spikedriven/github_version/conf/tiny_imagenet/6_512_300E_t4_from_cifar10.yml \
# --model sdt \
# --output /zhengzeqi/top_down/spikedriven/github_version/weights \
# --time-steps 2 \
# --batch-size 128 \
# --val-batch-size 128 \
# --recurrent_coding \
# --pe_type 3d_pe_arch_1 \
# --experiment tiny_imagenet_T_2_3d_pe_arch_1



# python -m torch.distributed.launch --nproc_per_node=4 /zhengzeqi/top_down/spikedriven/github_version/train.py \
# -c /zhengzeqi/top_down/spikedriven/github_version/conf/tiny_imagenet/6_512_300E_t4_from_cifar10.yml \
# --model sdt \
# --output /zhengzeqi/top_down/spikedriven/github_version/weights \
# --time-steps 2 \
# --batch-size 128 \
# --val-batch-size 128 \
# --recurrent_coding \
# --pe_type 3d_pe_arch_2 \
# --experiment tiny_imagenet_T_2_3d_pe_arch_2




# python -m torch.distributed.launch --nproc_per_node=4 /zhengzeqi/top_down/spikedriven/github_version/train.py \
# -c /zhengzeqi/top_down/spikedriven/github_version/conf/tiny_imagenet/6_512_300E_t4_from_cifar10.yml \
# --model sdt \
# --output /zhengzeqi/top_down/spikedriven/github_version/weights \
# --time-steps 2 \
# --batch-size 128 \
# --val-batch-size 128 \
# --recurrent_coding \
# --pe_type 3d_pe_arch_3 \
# --experiment tiny_imagenet_T_2_3d_pe_arch_3



# python -m torch.distributed.launch --nproc_per_node=4 /zhengzeqi/top_down/spikedriven/github_version/train.py \
# -c /zhengzeqi/top_down/spikedriven/github_version/conf/tiny_imagenet/6_512_300E_t4_from_cifar10.yml \
# --model sdt \
# --output /zhengzeqi/top_down/spikedriven/github_version/weights \
# --time-steps 2 \
# --batch-size 128 \
# --val-batch-size 128 \
# --recurrent_coding \
# --pe_type 3d_pe_arch_4 \
# --experiment tiny_imagenet_T_2_3d_pe_arch_4



python -m torch.distributed.launch --nproc_per_node=4 /zhengzeqi/top_down/spikedriven/github_version/train.py \
-c /zhengzeqi/top_down/spikedriven/github_version/conf/tiny_imagenet/6_512_300E_t4_from_cifar10.yml \
--model sdt \
--output /zhengzeqi/top_down/spikedriven/github_version/weights \
--time-steps 6 \
--batch-size 128 \
--val-batch-size 128 \
--recurrent_coding \
--pe_type 3d_pe_arch_5 \
--experiment tiny_imagenet_T_6_3d_pe_arch_5


