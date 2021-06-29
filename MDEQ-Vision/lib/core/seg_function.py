# Modified based on the HRNet repo.

import logging
import os
import time
import numpy as np
import sys

import numpy as np
import numpy.ma as ma
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn import functional as F

from utils.utils import AverageMeter
from utils.utils import get_confusion_matrix
from utils.utils import adjust_learning_rate
from utils.utils import get_world_size, get_rank
from utils.utils import save_checkpoint

logger = logging.getLogger(__name__)

def reduce_tensor(inp):
    """
    Reduce the loss from all processes so that 
    process with rank 0 has the averaged results.
    """
    world_size = get_world_size()
    if world_size < 2:
        return inp
    with torch.no_grad():
        reduced_inp = inp
        dist.reduce(reduced_inp, dst=0)
    return reduced_inp

def train(config, epoch, num_epoch, epoch_iters, base_lr, num_iters,
         trainloader, optimizer, lr_scheduler, model, output_dir, writer_dict, device):
    
    # Training
    model.train()
    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    ave_jac_loss = AverageMeter()
    tic = time.time()
    cur_iters = epoch*epoch_iters
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']
    assert global_steps == cur_iters, "Step counter problem... fix this?"
    update_freq = config.LOSS.JAC_INCREMENTAL

    # Distributed information
    rank = get_rank()
    world_size = get_world_size()

    for i_iter, batch in enumerate(trainloader):
        images, labels, _, _ = batch
        images = images.to(device)
        labels = labels.long().to(device)

        # compute output
        deq_steps = global_steps - config.TRAIN.PRETRAIN_STEPS
        if deq_steps < 0:
            factor = config.LOSS.PRETRAIN_JAC_LOSS_WEIGHT
        elif config.LOSS.JAC_STOP_EPOCH <= epoch:
            # If are above certain epoch, we may want to stop jacobian regularization training
            # (e.g., when the original loss is 0.01 and jac loss is 0.05, the jacobian regularization
            # will be dominating and hurt performance!)
            factor = 0
        else:
            factor = config.LOSS.JAC_LOSS_WEIGHT + 0.1 * (deq_steps // update_freq)
        compute_jac_loss = (np.random.uniform(0,1) < config.LOSS.JAC_LOSS_FREQ) and (factor > 0)
        delta_f_thres = random.randint(-config.DEQ.RAND_F_THRES_DELTA,1) if (config.DEQ.RAND_F_THRES_DELTA > 0 and compute_jac_loss) else 0
        f_thres = config.DEQ.F_THRES + delta_f_thres
        b_thres = config.DEQ.B_THRES
        losses, jac_loss, _, _ = model(images, labels, train_step=global_steps, 
                                       compute_jac_loss=compute_jac_loss,
                                       f_thres=f_thres, b_thres=b_thres, writer=writer)
        loss = losses.mean()
        jac_loss = jac_loss.mean()

        reduced_loss = reduce_tensor(loss)
        reduced_jac_loss = reduce_tensor(jac_loss) if compute_jac_loss else jac_loss

        # compute gradient and do update step
        model.zero_grad()
        if factor > 0:
            (loss + factor*jac_loss).backward()
        else:
            loss.backward()
        if config.TRAIN.CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP)
        optimizer.step()
        if config.TRAIN.LR_SCHEDULER == 'cosine':
            lr_scheduler.step()
        else:
            # If LR scheduler is None
            lr = adjust_learning_rate(optimizer, base_lr, num_iters, i_iter+cur_iters)
        
        # update average loss
        ave_loss.update(reduced_loss.item())
        if compute_jac_loss:
            ave_jac_loss.update(reduced_jac_loss.item())

        # measure elapsed time (modeling + data + sync)
        batch_time.update(time.time() - tic)
        tic = time.time()

        if i_iter % config.PRINT_FREQ == 0 and rank == 0:
            print_loss = ave_loss.average() / world_size
            print_jac_loss = ave_jac_loss.average() / world_size
            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                  'lr: {:.6f}, Loss: {:.6f}, Jac: {:.4f} ({:.4f})' .format(
                      epoch, num_epoch, i_iter, epoch_iters, 
                      batch_time.average(), lr, print_loss, print_jac_loss, factor)
            logging.info(msg)

        global_steps += 1
        writer_dict['train_global_steps'] = global_steps

        if factor > 0 and global_steps > config.TRAIN.PRETRAIN_STEPS and deq_steps % update_freq == 0:
             logger.info(f'Note: Adding 0.1 to Jacobian regularization weight.')


def validate(config, testloader, model, lr_scheduler, epoch, writer_dict, device, 
             spectral_radius_mode=False):
    model.eval()
    ave_loss = AverageMeter()
    if spectral_radius_mode:
        ave_sradius = AverageMeter()
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']
    confusion_matrix = np.zeros((config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))

    # Distributed information
    rank = get_rank()
    world_size = get_world_size()

    with torch.no_grad():
        for _, batch in enumerate(testloader):
            image, label, _, _ = batch
            size = label.size()
            image = image.to(device)
            label = label.long().to(device)

            losses, _, pred, _ = model(image, label, train_step=(-1 if epoch < 0 else global_steps), 
                                       compute_jac_loss=False, spectral_radius_mode=spectral_radius_mode,
                                       writer=writer)
            pred = F.interpolate(input=pred, size=(size[-2], size[-1]), mode='bilinear', align_corners=True)
            loss = losses.mean()

            # record loss
            reduced_loss = reduce_tensor(loss)
            ave_loss.update(reduced_loss.item())
            confusion_matrix += get_confusion_matrix(
                label,
                pred,
                size,
                config.DATASET.NUM_CLASSES,
                config.TRAIN.IGNORE_LABEL)
            
            if spectral_radius_mode:
                sradius = sradius.mean()
                ave_sradius.update(sradius.item(), input.size(0))

    confusion_matrix = torch.from_numpy(confusion_matrix).to(device)
    reduced_confusion_matrix = reduce_tensor(confusion_matrix)

    confusion_matrix = reduced_confusion_matrix.cpu().numpy()
    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()
    print_loss = ave_loss.average()/world_size

    if rank == 0:
        if spectral_radius_mode:
            logger.info(f"Spectral radius over validation set: {sradiuses.average()}")
    return print_loss, mean_IoU, IoU_array
    

def testval(config, test_dataset, testloader, model, sv_dir='', sv_pred=False):
    model.eval()
    confusion_matrix = np.zeros((config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))
    with torch.no_grad():
        for index, batch in enumerate(tqdm(testloader)):
            image, label, _, name = batch
            size = label.size()
            pred = test_dataset.multi_scale_inference(model, image, 
                        scales=config.TEST.SCALE_LIST, 
                        flip=config.TEST.FLIP_TEST)
            
            if pred.size()[-2] != size[-2] or pred.size()[-1] != size[-1]:
                pred = F.interpolate(pred, (size[-2], size[-1]), mode='bilinear', align_corners=True)

            confusion_matrix += get_confusion_matrix(
                label,
                pred,
                size,
                config.DATASET.NUM_CLASSES,
                config.TRAIN.IGNORE_LABEL)

            if sv_pred:
                sv_path = os.path.join(sv_dir,'test_val_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred(pred, sv_path, name)
            
            if index % 100 == 0:
                logging.info('processing: %d images' % index)
                pos = confusion_matrix.sum(1)
                res = confusion_matrix.sum(0)
                tp = np.diag(confusion_matrix)
                IoU_array = (tp / np.maximum(1.0, pos + res - tp))
                mean_IoU = IoU_array.mean()
                logging.info('mIoU: %.4f' % (mean_IoU))

    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    pixel_acc = tp.sum()/pos.sum()
    mean_acc = (tp/np.maximum(1.0, pos)).mean()
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()

    return mean_IoU, IoU_array, pixel_acc, mean_acc

def test(config, test_dataset, testloader, model, sv_dir='', sv_pred=True):
    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(tqdm(testloader)):
            image, size, name = batch
            size = size[0]
            pred = test_dataset.multi_scale_inference(model, image, 
                        scales=config.TEST.SCALE_LIST, 
                        flip=config.TEST.FLIP_TEST)
            
            if pred.size()[-2] != size[0] or pred.size()[-1] != size[1]:
                pred = F.interpolate(pred, (size[-2], size[-1]), mode='bilinear', align_corners=True)

            if sv_pred:
                sv_path = os.path.join(sv_dir,'test_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred(pred, sv_path, name)
