# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import timm

# assert timm.__version__ == "0.3.2"  # version check
from timm.models.layers import trunc_normal_
import timm.optim.optim_factory as optim_factory
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import util.lr_decay_hst as lrd
import util.misc as misc
from util.datasets import build_dataset
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import qkformer

from engine_finetune import train_one_epoch, evaluate, evaluate_return_ETF_LIF_with_firing_rate

from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler

def get_args_parser():
    # important params
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--data_path', default='/media/data/imagenet2012', type=str,
                        help='dataset path')

    # Model parameters
    parser.add_argument('--model', default='QKFormer_10_384', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--time_step', default=4, type=int,
                        help='images input size')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    #add in 2025-04-29
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "adamw"')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='Optimizer momentum (default: 0.9)')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    # lr
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=6e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=1.0,
                        help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')
    
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR', 
                        help='lower lr bound for cyclic schedulers that hit 0')             # 2025-04-29
    
    parser.add_argument('--eval_metric', default='acc1', type=str, metavar='EVAL_METRIC',
                    help='Best metric (default: "acc1"')                                    # 2025-04-29
    
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "step"')                                   # 2025-04-29
    
    parser.add_argument('--cooldown-epochs', type=int, default=5, metavar='N',              # 2025-04-29
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params

    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # Dataset parameters

    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')

    parser.add_argument('--output_dir', default='./output_dir_qkformer',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir_qkformer',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    
    #changed on 2025-04-22
    parser.add_argument('--recurrent_coding', action='store_true', default=False)
    parser.add_argument('--recurrent_lif', type=str, default=None, help="lif, plif, or None")
    parser.add_argument('--pe_type', default=None, help="position embedding methods")

    parser.add_argument('--temporal_conv_type', default=None, help="temporal feedback conv type")

    #changed on 2025-05-06
    # 是否启用IMP神经元
    parser.add_argument('--use_imp_lif',  action='store_true', default=False,
                        help="use initial membrance potential LIF")

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_train = build_dataset(is_train=True, args=args)
    dataset_val = build_dataset(is_train=False, args=args)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank,
                shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    # changed on 2025-04-29
    if args.finetune:
        model = qkformer.__dict__[args.model](T=args.time_step,finetune = args.finetune, recurrent_coding=args.recurrent_coding, 
                                            recurrent_lif=args.recurrent_lif, pe_type=args.pe_type, temporal_conv_type=args.temporal_conv_type,
                                            input_H=args.input_size, input_W=args.input_size, use_imp_lif=args.use_imp_lif)
    else:
        model = qkformer.__dict__[args.model](T=args.time_step, recurrent_coding=args.recurrent_coding, recurrent_lif=args.recurrent_lif, 
                                            pe_type=args.pe_type, temporal_conv_type=args.temporal_conv_type,
                                            input_H=args.input_size, input_W=args.input_size, use_imp_lif=args.use_imp_lif)


    # if args.finetune and not args.eval:
    #     checkpoint = torch.load(args.finetune, map_location='cpu')

    #     print("Load pre-trained checkpoint from: %s" % args.finetune)
    #     checkpoint_model = checkpoint['model']
    #     state_dict = model.state_dict()
    #     for k in ['head.weight', 'head.bias']:
    #         if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
    #             print(f"Removing key {k} from pretrained checkpoint")
    #             del checkpoint_model[k]

    #     # interpolate position embedding
    #     # interpolate_pos_embed(model, checkpoint_model)

    #     # load pre-trained model
    #     msg = model.load_state_dict(checkpoint_model, strict=False)
    #     print(msg)

    #     # if args.global_pool:
    #     #     assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
    #     # else:
    #     #     assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

    #     # manually initialize fc layer
    #     trunc_normal_(model.head.weight, std=2e-5)

    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    # if args.distributed:
    #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
    #     model_without_ddp = model.module
    # changed on 2025-04-29
    if True:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], 
                                                          find_unused_parameters=False,broadcast_buffers=False)
        model_without_ddp = model.module


    # build optimizer with layer-wise lr decay (lrd)
    # param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
    #                                     # no_weight_decay_list=model_without_ddp.no_weight_decay(),
    #                                     layer_decay=args.layer_decay
    #                                     )
    # optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    # loss_scaler = NativeScaler()

    # changed on 2025-04-29, epoch_wise lr decay
    loss_scaler = NativeScaler() 
    optimizer = create_optimizer_v2(model_without_ddp, **optimizer_kwargs(cfg=args))
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)


    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    if args.resume:
        misc.load_model_noopt(args=args, model_without_ddp=model_without_ddp)
    else:
        misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        # test_stats = evaluate_return_ETF_LIF_with_firing_rate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
            
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, mixup_fn,
            log_writer=log_writer,
            lr_scheduler=lr_scheduler,   # changed on 2025-04-29
            args=args
        )
        
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")

        if lr_scheduler is not None:
            lr_scheduler.step(epoch + 1, test_stats[args.eval_metric]) #ADD 2024/11/14)

        # if epoch > int(args.epochs - 20) and test_stats["acc1"] > max_accuracy:
        if True:
            if args.output_dir:
                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch)

        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f'Max accuracy: {max_accuracy:.2f}%')

        if log_writer is not None:
            log_writer.add_scalar('perf/test_acc1', test_stats['acc1'], epoch)
            log_writer.add_scalar('perf/test_acc5', test_stats['acc5'], epoch)
            log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
