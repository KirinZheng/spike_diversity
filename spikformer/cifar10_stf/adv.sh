CUDA_VISIBLE_DEVICES=2 python train_d.py \
  --adv PGD \
  --eval \
  -T 4\
  --weight_dir /zhuzizheng/code/td_reset/spikformer-main/cifar/output/cifar100/T4/recon/0/model_best.pth.tar\
  -c ./cifar10.yml\
  --layer_td 'batch' \
  --td
  # -vb 100
  # -c ./cifar100.yml\

  # -L 2\
  # -vb 124
  # -c \
  # --op 2 
  # --pretrained \
    # --eps 0.3 \v