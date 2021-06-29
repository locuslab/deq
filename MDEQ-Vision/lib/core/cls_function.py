# Modified based on the HRNet repo.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import numpy as np
import sys

import torch

from core.cls_evaluate import accuracy
sys.path.append("../")
from utils.utils import save_checkpoint, AverageMeter
import random
from tqdm import tqdm


logger = logging.getLogger(__name__)

        
def train(config, train_loader, model, criterion, optimizer, lr_scheduler, epoch,
          output_dir, tb_log_dir, writer_dict=None, topk=(1,5)):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    jac_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    writer = writer_dict['writer'] if writer_dict else None
    global_steps = writer_dict['train_global_steps']
    update_freq = config.LOSS.JAC_INCREMENTAL

    # switch to train mode
    model.train()

    end = time.time()
    total_batch_num = len(train_loader)
    effec_batch_num = int(config.PERCENT * total_batch_num)
    for i, (input, target) in enumerate(train_loader):
        # train on partial training data
        if i >= effec_batch_num: break
            
        # measure data loading time
        data_time.update(time.time() - end)

        # compute jacobian loss weight (which is dynamically scheduled)
        deq_steps = global_steps - config.TRAIN.PRETRAIN_STEPS
        if deq_steps < 0:
            # We can also regularize output Jacobian when pretraining
            factor = config.LOSS.PRETRAIN_JAC_LOSS_WEIGHT
        elif epoch >= config.LOSS.JAC_STOP_EPOCH:
            # If are above certain epoch, we may want to stop jacobian regularization training
            # (e.g., when the original loss is 0.01 and jac loss is 0.05, the jacobian regularization
            # will be dominating and hurt performance!)
            factor = 0
        else:
            # Dynamically schedule the Jacobian reguarlization loss weight, if needed
            factor = config.LOSS.JAC_LOSS_WEIGHT + 0.1 * (deq_steps // update_freq)
        compute_jac_loss = (torch.rand([]).item() < config.LOSS.JAC_LOSS_FREQ) and (factor > 0)
        delta_f_thres = torch.randint(-config.DEQ.RAND_F_THRES_DELTA,2,[]).item() if (config.DEQ.RAND_F_THRES_DELTA > 0 and compute_jac_loss) else 0
        f_thres = config.DEQ.F_THRES + delta_f_thres
        b_thres = config.DEQ.B_THRES
        output, jac_loss, _ = model(input, train_step=(lr_scheduler._step_count-1), 
                                    compute_jac_loss=compute_jac_loss,
                                    f_thres=f_thres, b_thres=b_thres, writer=writer)
        target = target.cuda(non_blocking=True)
        loss = criterion(output, target)
        jac_loss = jac_loss.mean()

        # compute gradient and do update step
        optimizer.zero_grad()
        if factor > 0:
            (loss + factor*jac_loss).backward()
        else:
            loss.backward()
        if config.TRAIN.CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP)
        optimizer.step()
        if config.TRAIN.LR_SCHEDULER != 'step':
            lr_scheduler.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))
        if compute_jac_loss:
            jac_losses.update(jac_loss.item(), input.size(0))

        prec1, prec5 = accuracy(output, target, topk=topk)
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}] ({3})\t' \
                  'Time {batch_time.avg:.3f}s\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.avg:.3f}s\t' \
                  'Loss {loss.avg:.5f}\t' \
                  'Jac (gamma) {jac_losses.avg:.4f} ({factor:.4f})\t' \
                  'Acc@1 {top1.avg:.3f}\t'.format(
                      epoch, i, effec_batch_num, global_steps, batch_time=batch_time,
                      speed=input.size(0)/batch_time.avg,
                      data_time=data_time, loss=losses, jac_losses=jac_losses, factor=factor, top1=top1)
            if 5 in topk:
                msg += 'Acc@5 {top5.avg:.3f}\t'.format(top5=top5)
            logger.info(msg)
            
        global_steps += 1
        writer_dict['train_global_steps'] = global_steps
        
        if factor > 0 and global_steps > config.TRAIN.PRETRAIN_STEPS and (deq_steps+1) % update_freq == 0:
             logger.info(f'Note: Adding 0.1 to Jacobian regularization weight.')


def validate(config, val_loader, model, criterion, lr_scheduler, epoch, output_dir, tb_log_dir,
             writer_dict=None, topk=(1,5), spectral_radius_mode=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    spectral_radius_mode = spectral_radius_mode and (epoch % 10 == 0)
    if spectral_radius_mode:
        sradiuses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    writer = writer_dict['writer'] if writer_dict else None

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        # tk0 = tqdm(enumerate(val_loader), total=len(val_loader), position=0, leave=True)
        for i, (input, target) in enumerate(val_loader):
            # compute output
            output, _, sradius = model(input, 
                                 train_step=(-1 if epoch < 0 else (lr_scheduler._step_count-1)),
                                 compute_jac_loss=False, spectral_radius_mode=spectral_radius_mode,
                                 writer=writer)
            target = target.cuda(non_blocking=True)
            loss = criterion(output, target)

            # measure accuracy and record loss
            losses.update(loss.item(), input.size(0))
            prec1, prec5 = accuracy(output, target, topk=topk)
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            if spectral_radius_mode:
                sradius = sradius.mean()
                sradiuses.update(sradius.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    if spectral_radius_mode:
        logger.info(f"Spectral radius over validation set: {sradiuses.avg}")    
    msg = 'Test: Time {batch_time.avg:.3f}\t' \
            'Loss {loss.avg:.4f}\t' \
            'Acc@1 {top1.avg:.3f}\t'.format(
                batch_time=batch_time, loss=losses, top1=top1)
    if 5 in topk:
        msg += 'Acc@5 {top5.avg:.3f}\t'.format(top5=top5)
    logger.info(msg)

    if writer:
        writer.add_scalar('accuracy/valid_top1', top1.avg, epoch)
        if spectral_radius_mode:
            writer.add_scalar('stability/sradius', sradiuses.avg, epoch)

    return top1.avg
