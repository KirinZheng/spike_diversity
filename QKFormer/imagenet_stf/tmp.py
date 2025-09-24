import torch
import argparse

parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
parser.add_argument('--ckp_path', type=str, default = None)

args = parser.parse_args()

tmp = torch.load(args.ckp_path, map_location="cpu")

ckp_args_dict = vars(tmp['args'])

key_list = ['model', 'input_size', 'time_step', 'accum_iter', 'finetune',
            'resume', 'warmup_epochs', 'warmup_lr', 'batch_size',
            'beta', 'blr', 'epochs', 'lr', 'min_lr']

for element in key_list:
    print(f"{element}: {ckp_args_dict[element]}")