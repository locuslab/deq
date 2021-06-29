# Modified based on the HRNet repo.

import argparse
import os
import pprint
import shutil
import sys

import logging
import time
import timeit
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

import _init_paths
import models
import datasets
from config import config
from config import update_config
from core.seg_criterion import CrossEntropy, OhemCrossEntropy
from core.seg_function import train, validate
from utils.modelsummary import get_model_summary
from utils.utils import create_logger, FullModel, get_rank, get_optimizer
from termcolor import colored

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--testModel',
                        help='testModel',
                        type=str,
                        default='')
    parser.add_argument('--percent',
                        help='percentage of training data to use',
                        type=float,
                        default=1.0)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args

def main():
    args = parse_args()
    
    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(config)

    writer_dict = {
        'writer': SummaryWriter(tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED
    gpus = list(config.GPUS)
    distributed = len(gpus) > 1
    device = torch.device('cuda:{}'.format(args.local_rank))

    # build model
    model = eval('models.'+config.MODEL.NAME +
                 '.get_seg_net')(config)
    
    if config.TRAIN.MODEL_FILE:
        model_state_file = config.TRAIN.MODEL_FILE
        logger.info(colored('=> loading model from {}'.format(model_state_file), 'red'))
        
        pretrained_dict = torch.load(model_state_file)
        model_dict = model.state_dict()
        pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                           if k[6:] in model_dict.keys()}      # To remove the "model." from state dict
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    
    if args.local_rank == 0:
        # copy model file
        this_dir = os.path.dirname(__file__)
        models_dst_dir = os.path.join(final_output_dir, 'models')
        if os.path.exists(models_dst_dir):
            shutil.rmtree(models_dst_dir)
        shutil.copytree(os.path.join(this_dir, '../lib/models'), models_dst_dir)

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://",
        )

    # prepare data
    crop_size = (config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    train_dataset = eval('datasets.'+config.DATASET.DATASET)(
                        root=config.DATASET.ROOT,
                        list_path=config.DATASET.TRAIN_SET,
                        num_samples=None,
                        num_classes=config.DATASET.NUM_CLASSES,
                        multi_scale=config.TRAIN.MULTI_SCALE,
                        flip=config.TRAIN.FLIP,
                        ignore_label=config.TRAIN.IGNORE_LABEL,
                        base_size=config.TRAIN.BASE_SIZE,
                        crop_size=crop_size,
                        downsample_rate=config.TRAIN.DOWNSAMPLERATE,
                        scale_factor=config.TRAIN.SCALE_FACTOR)
    
    if distributed:
        train_sampler = DistributedSampler(train_dataset)
    else:
        train_sampler = None

    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU,
        shuffle=config.TRAIN.SHUFFLE and train_sampler is None,
        num_workers=config.WORKERS,
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler)

    if config.DATASET.EXTRA_TRAIN_SET:
        extra_train_dataset = eval('datasets.'+config.DATASET.DATASET)(
                    root=config.DATASET.ROOT,
                    list_path=config.DATASET.EXTRA_TRAIN_SET,
                    num_samples=None,
                    num_classes=config.DATASET.NUM_CLASSES,
                    multi_scale=config.TRAIN.MULTI_SCALE,
                    flip=config.TRAIN.FLIP,
                    ignore_label=config.TRAIN.IGNORE_LABEL,
                    base_size=config.TRAIN.BASE_SIZE,
                    crop_size=crop_size,
                    downsample_rate=config.TRAIN.DOWNSAMPLERATE,
                    scale_factor=config.TRAIN.SCALE_FACTOR)

        if distributed:
            extra_train_sampler = DistributedSampler(extra_train_dataset)
        else:
            extra_train_sampler = None

        extra_trainloader = torch.utils.data.DataLoader(
            extra_train_dataset,
            batch_size=config.TRAIN.BATCH_SIZE_PER_GPU,
            shuffle=config.TRAIN.SHUFFLE and extra_train_sampler is None,
            num_workers=config.WORKERS,
            pin_memory=True,
            drop_last=True,
            sampler=extra_train_sampler)

    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    test_dataset = eval('datasets.'+config.DATASET.DATASET)(
                        root=config.DATASET.ROOT,
                        list_path=config.DATASET.TEST_SET,
                        num_samples=config.TEST.NUM_SAMPLES,
                        num_classes=config.DATASET.NUM_CLASSES,
                        multi_scale=False,
                        flip=False,
                        ignore_label=config.TRAIN.IGNORE_LABEL,
                        base_size=config.TEST.BASE_SIZE,
                        crop_size=test_size,
                        center_crop_test=config.TEST.CENTER_CROP_TEST,
                        downsample_rate=1)

    if distributed:
        test_sampler = DistributedSampler(test_dataset)
    else:
        test_sampler = None

    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True,
        sampler=test_sampler)
    
    # criterion
    if config.LOSS.USE_OHEM:
        criterion = OhemCrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                     thres=config.LOSS.OHEMTHRES,
                                     min_kept=config.LOSS.OHEMKEEP,
                                     weight=train_dataset.class_weights)
    else:
        criterion = CrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                 weight=train_dataset.class_weights)

    model = FullModel(model, criterion)
    model = model.to(device)
    if distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

    # optimizer
    optimizer = get_optimizer(config, model)
    epoch_iters = np.int(train_dataset.__len__() / config.TRAIN.BATCH_SIZE_PER_GPU / len(gpus))
    end_epoch = config.TRAIN.END_EPOCH + config.TRAIN.EXTRA_EPOCH
    num_iters = config.TRAIN.END_EPOCH * epoch_iters
    extra_iters = config.TRAIN.EXTRA_EPOCH * epoch_iters
    best_mIoU = 0
    last_epoch = 0
    lr_scheduler = None
    if config.TRAIN.RESUME:
        model_state_file = os.path.join(final_output_dir, 'checkpoint.pth.tar')
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file, 
                        map_location=lambda storage, loc: storage)
            best_mIoU = checkpoint['best_mIoU']
            last_epoch = checkpoint['epoch']
            writer_dict['train_global_steps'] = checkpoint['writer_dict']['train_global_steps']
            writer_dict['valid_global_steps'] = checkpoint['writer_dict']['valid_global_steps']
            model.module.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            
            if 'lr_scheduler' in checkpoint:
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, num_iters, eta_min=1e-6)
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            logger.info("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
    
    if lr_scheduler is None:
        if config.TRAIN.LR_SCHEDULER == 'cosine':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, num_iters, eta_min=1e-6)
        else:
            logger.info("Note: No scheduler specified. Use the default scheduler! (See utils/utils.py)")
            lr_scheduler = None

    start = timeit.default_timer()
    
    for epoch in range(last_epoch, end_epoch):
        
        if distributed:
            train_sampler.set_epoch(epoch)
        if epoch >= config.TRAIN.END_EPOCH:
            # Extra training dataset (shouldn't matter to DEQ models... yet)
            train(config, epoch-config.TRAIN.END_EPOCH, 
                  config.TRAIN.EXTRA_EPOCH, epoch_iters, 
                  config.TRAIN.EXTRA_LR, extra_iters, 
                  extra_trainloader, optimizer, lr_scheduler, model, 
                  final_output_dir, writer_dict, device)
        else:
            train(config, epoch, config.TRAIN.END_EPOCH, 
                  epoch_iters, config.TRAIN.LR, num_iters,
                  trainloader, optimizer, lr_scheduler, model, final_output_dir, writer_dict,
                  device)
            
        torch.cuda.empty_cache()
        valid_loss, mean_IoU, IoU_array = validate(config, testloader, model, lr_scheduler, epoch, writer_dict, device, 
                                                   spectral_radius_mode=config.DEQ.SPECTRAL_RADIUS_MODE)
        torch.cuda.empty_cache()
        writer_dict['writer'].flush()

        if args.local_rank == 0:
            # Only the master node is responsible for saving models and logging
            logger.info('=> saving checkpoint to {}'.format(final_output_dir + 'checkpoint.pth.tar'))
            torch.save({
                'epoch': epoch+1,
                'best_mIoU': best_mIoU,
                'state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
            }, os.path.join(final_output_dir,'checkpoint.pth.tar'))

            if mean_IoU > best_mIoU:
                best_mIoU = mean_IoU
                torch.save(model.module.state_dict(), os.path.join(final_output_dir, 'best.pth'))
            msg = 'Loss: {:.3f}, MeanIU: {: 4.4f}, Best_mIoU: {: 4.4f}'.format(valid_loss, mean_IoU, best_mIoU)
            logger.info(msg)
            logger.info(IoU_array)

            if epoch == end_epoch - 1:
                final_model_state_file = os.path.join(final_output_dir, 'final_state.pth')
                logger.info('saving final model state to {}'.format(final_model_state_file))
                torch.save(model.module.state_dict(), final_model_state_file)
                writer_dict['writer'].close()
                
                end = timeit.default_timer()
                logger.info('Hours: %d' % np.int((end-start)/3600))
                logger.info('Done')


if __name__ == '__main__':
    main()
