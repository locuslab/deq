# coding: utf-8
import argparse
import time
import math
import os, sys
import itertools
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from data_utils import get_lm_corpus
from models.trellisnets.deq_trellisnet import DEQTrellisNetLM
from modules import radam
from utils.exp_utils import create_exp_dir
from utils.data_parallel import BalancedDataParallel
from utils.splitcross import *


parser = argparse.ArgumentParser(description='PyTorch DEQ Sequence Model')
parser.add_argument('--data', type=str, default='../data/wikitext-103',
                    help='location of the data corpus (default to the WT103 path)')
parser.add_argument('--dataset', type=str, default='wt103',
                    choices=['wt103'],
                    help='dataset name')
parser.add_argument('--n_layer', type=int, default=12,
                    help='number of total layers')
parser.add_argument('--d_embed', type=int, default=500,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=1400,
                    help='number of hidden units per layer')
parser.add_argument('--nout', type=int, default=500,
                    help='number of output units')
parser.add_argument('--epochs', type=int, default=25,
                    help='upper epoch limit (default: 25)')

# Optimizers
parser.add_argument('--optim', default='Adam', type=str,
                    choices=['Adam', 'SGD', 'Adagrad', 'RMSprop', 'RAdam'],
                    help='optimizer to use.')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate (default: 1e-3)')

# Gradient updates
parser.add_argument('--clip', type=float, default=0.07,
                    help='gradient clipping (default: 0.07)')
parser.add_argument('--batch_size', type=int, default=60,
                    help='batch size')
parser.add_argument('--batch_chunk', type=int, default=1,
                    help='split batch into chunks to save memory')

# Sequence logistics
parser.add_argument('--seq_len', type=int, default=100,
                    help='total sequence length')
parser.add_argument('--subseq_len', type=int, default=50,
                    help='length of subsequence processed each time by DEQ')

# Regularizations
parser.add_argument('--dropout', type=float, default=0.1,
                    help='output dropout (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.1,
                    help='input dropout (0 = no dropout)')
parser.add_argument('--wdrop', type=float, default=0.0,
                    help='dropout applied to weights (0 = no dropout)')
parser.add_argument('--emb_dropout', type=float, default=0.0,
                    help='dropout applied to embedding layer (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.1,
                    help='dropout applied to hidden layers (0 = no dropout)')
parser.add_argument('--weight_decay', type=float, default=0.0,
                    help='weight decay')
parser.add_argument('--wnorm', action='store_false',
                    help='use weight normalization (default: True)')

# Training techniques
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--not_tied', action='store_true',
                    help='do not tie the word embedding and softmax weights (default: False)')
parser.add_argument('--anneal', type=int, default=5,
                    help='learning rate annealing criteria (default: 5)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval')
parser.add_argument('--when', nargs='+', type=int, default=[15, 20, 23],
                    help='When to decay the learning rate')
parser.add_argument('--ksize', type=int, default=2,
                    help='conv kernel size (default: 2)')
parser.add_argument('--dilation', type=int, default=1,
                    help='dilation rate (default: 1)')
parser.add_argument('--n_experts', type=int, default=0,
                    help='number of softmax experts (default: 0)')
parser.add_argument('--multi_gpu', action='store_true',
                    help='use multiple GPU')
parser.add_argument('--f_thres', type=int, default=50,
                    help='forward pass Broyden threshold')
parser.add_argument('--b_thres', type=int, default=80,
                    help='backward pass Broyden threshold')
parser.add_argument('--work_dir', default='LM-TRE', type=str,
                    help='experiment directory.')
parser.add_argument('--restart', action='store_true',
                    help='restart training from the saved checkpoint')
parser.add_argument('--restart_dir', type=str, default='',
                    help='restart dir')
parser.add_argument('--debug', action='store_true',
                    help='run in debug mode (do not create exp dir)')
parser.add_argument('--gpu0_bsz', type=int, default=-1,
                    help='batch size on gpu 0')
parser.add_argument('--pretrain_steps', type=int, default=0,
                    help='number of pretrain steps')
parser.add_argument('--start_train_steps', type=int, default=0,
                    help='starting training step count (default to 0)')
parser.add_argument('--eval', action='store_true',
                    help='evaluation mode')
parser.add_argument('--load', type=str, default='',
                    help='path to load weight')
parser.add_argument('--name', type=str, default='N/A',
                    help='name of the trial')

args = parser.parse_args()
args.tied = not args.not_tied
args.pretrain_steps += args.start_train_steps
print(f"Experiment name: {args.name}")
assert args.seq_len > 0, "For now you must set seq_len > 0 when using deq"
args.work_dir += "deq"
args.cuda = torch.cuda.is_available()
    
if args.d_embed < 0:
    args.d_embed = args.nout

assert args.batch_size % args.batch_chunk == 0

args.work_dir = '{}-{}'.format(args.work_dir, args.dataset)
args.work_dir = os.path.join(args.work_dir, time.strftime('%Y%m%d-%H%M%S'))
logging = create_exp_dir(args.work_dir,
    scripts_to_save=['train_trellisnet.py', 'models/trellisnets/deq_trellisnet.py'], debug=args.debug)

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print('WARNING: You have a CUDA device, so you should probably run with --cuda')
    else:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.cuda.manual_seed_all(args.seed)

device = torch.device('cuda' if args.cuda else 'cpu')

###############################################################################
# Load data
###############################################################################
corpus = get_lm_corpus(args.data, args.dataset)
ntokens = len(corpus.vocab)
args.n_token = ntokens

eval_batch_size = 4
tr_iter = corpus.get_iterator('train', args.batch_size, args.seq_len, device=device)
va_iter = corpus.get_iterator('valid', eval_batch_size, args.seq_len, device=device)
te_iter = corpus.get_iterator('test', eval_batch_size, args.seq_len, device=device)

# adaptive softmax
cutoffs = [2800, 20000, 76000]    # This can be tuned.
criterion = SplitCrossEntropyLoss(args.d_embed, splits=cutoffs, verbose=False) 

model = DEQTrellisNetLM(n_token=ntokens, n_layer=args.n_layer, ninp=args.d_embed, nhid=args.nhid, nout=args.nout, 
                        kernel_size=args.ksize, emb_dropout=args.emb_dropout, dropouti=args.dropouti, dropout=args.dropout, 
                        dropouth=args.dropouth, wdrop=args.wdrop, wnorm=args.wnorm, tie_weights=args.tied, 
                        pretrain_steps=args.pretrain_steps, dilation=args.dilation, load=args.load)

args.n_all_param = sum([p.nelement() for p in model.parameters() if p.requires_grad]) + \
                   sum([p.nelement() for p in criterion.parameters() if p.requires_grad])

if args.multi_gpu:
    model = model.to(device)
    if args.gpu0_bsz >= 0:
        para_model = BalancedDataParallel(args.gpu0_bsz // args.batch_chunk, model, dim=1).to(device)   # Batch dim is dim 1
    else:
        para_model = nn.DataParallel(model, dim=1).to(device)
else:
    para_model = model.to(device)
    
    
#### optimizer
params = list(model.parameters()) + list(criterion.parameters())
lr = args.lr
optimizer = getattr(optim if args.optim != 'RAdam' else radam, args.optim)(params, lr=lr, weight_decay=args.weight_decay)

logging('=' * 100)
for k, v in args.__dict__.items():
    logging('    - {} : {}'.format(k, v))
logging('=' * 100)
logging(f'#params = {args.n_all_param}')

###############################################################################
# Training code
###############################################################################

def evaluate(eval_iter):
    global train_step
    model.eval()
    subseq_len = args.subseq_len

    # Evaluation
    total_len, total_loss = 0, 0.
    with torch.no_grad():
        mems = []
        for i, (data, target, seq_len) in enumerate(eval_iter):
            if mems:
                mems[0] = mems[0].detach()
            
            # output has dimension (seq_len x bsz x nout)
            (_, _, output), mems = model(data, target, mems, train_step=train_step, f_thres=args.f_thres, 
                                         b_thres=args.b_thres, subseq_len=subseq_len, decode=False)  
            loss = criterion(model.decoder.weight, model.decoder.bias, 
                             output.contiguous().view(-1, output.size(2)), target.view(-1))
            total_loss += seq_len * loss.item()
            total_len += seq_len
    
    model.train()
    return total_loss / total_len


def train():
    global train_step, log_start_time
    model.train()
    subseq_len = args.subseq_len

    train_loss = 0
    if args.batch_chunk > 1:
        mems = [[] for _ in range(args.batch_chunk)]  # Each chunk (apparent) should have its own memory padding
    else:
        mems = []

    for batch, (data, target, seq_len) in enumerate(tr_iter):
        optimizer.zero_grad()
        if args.batch_chunk > 1:
            # Mode 1: Using accumulated gradient to train on a larger (effective) batch size
            data_chunks = data.chunk(args.batch_chunk, dim=1)
            target_chunks = target.chunk(args.batch_chunk, dim=1)
            for i in range(args.batch_chunk):
                data_i = data_chunks[i].contiguous()
                target_i = target_chunks[i].contiguous()
                if mems[i]: mems[i][0] = mems[i][0].detach()
                (_, _, output_i), new_mem = para_model(data_i, target_i, mems[i], train_step=train_step, f_thres=args.f_thres, 
                                                       b_thres=args.b_thres, subseq_len=subseq_len, decode=False)
                loss = criterion(model.decoder.weight, model.decoder.bias, 
                                 output_i.view(-1, output_i.size(2)), target_i.view(-1))
                mems[i] = new_mem
                loss = loss / args.batch_chunk
                loss.backward()
                train_loss += loss.item()
                
        else:
            # Mode 2: Normal training with one batch per iteration
            if mems: mems[0] = mems[0].detach()
            (_, _, output), mems = para_model(data, target, mems, train_step=train_step, f_thres=args.f_thres, 
                                              b_thres=args.b_thres, subseq_len=subseq_len, decode=False)
            loss = criterion(model.decoder.weight, model.decoder.bias, 
                             output.reshape(-1, output.size(2)), target.view(-1))
            loss.backward()
            train_loss += loss.item()
            
        torch.nn.utils.clip_grad_norm_(params, args.clip)
        optimizer.step()
        train_step += 1
        
        # Logging of training progress
        if train_step % args.log_interval == 0:
            cur_loss = train_loss / args.log_interval
            elapsed = time.time() - log_start_time
            log_str = '| epoch {:3d} step {:>8d} | {:>6d} batches | lr {:.3g} ' \
                      '| ms/batch {:5.2f} | loss {:5.2f} | ppl {:9.3f}'.format(
                epoch, train_step, batch+1, optimizer.param_groups[0]['lr'],
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss))
            logging(log_str)
            train_loss = 0
            log_start_time = time.time()
    

# Loop over epochs.
train_step = args.start_train_steps
eval_count = 1
best_val_loss = None
all_val_losses = []

log_start_time = time.time()
eval_start_time = time.time()

if args.eval:
    epoch = -1
    valid_loss = evaluate(va_iter)
    logging('=' * 100)
    logging('| End of training | valid loss {:5.2f} | valid ppl {:9.3f}'.format(valid_loss, math.exp(valid_loss)))
    logging('=' * 100)
        
    test_loss = evaluate(te_iter)
    logging('=' * 100)
    logging('| End of training | test loss {:5.2f} | test ppl {:9.3f}'.format(test_loss, math.exp(test_loss)))
    logging('=' * 100)
    sys.exit(0)

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in itertools.count(start=1):
        train()
        val_loss = evaluate(va_iter)
        logging('-' * 100)
        log_str = '| Eval {:3d} at step {:>8d} | time: {:5.2f}s ' \
                  '| valid loss {:5.2f} | valid ppl {:9.3f}'.format(
            eval_count, train_step,
            (time.time() - eval_start_time), val_loss, math.exp(val_loss))
        logging(log_str)
        logging('-' * 100)
        
        eval_start_time = time.time()
        eval_count += 1
        
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            if not args.debug:
                with open(os.path.join(args.work_dir, 'model.pt'), 'wb') as f:
                    print(f'Saved Model! Experiment name: {args.name}')
                    torch.save(model, f)
                    model.save_weights(args.name)
                with open(os.path.join(args.work_dir, 'optimizer.pt'), 'wb') as f:
                    torch.save(optimizer.state_dict(), f)
            best_val_loss = val_loss
        
        if (len(all_val_losses) > args.anneal and val_loss > min(all_val_losses[:-args.anneal])) \
                or epoch in args.when:
            print("\n" + "*" * 89)
            if lr > 1e-5:
                print('Annealing learning rate')
                lr /= 10.0
                optimizer.param_groups[0]['lr'] = lr
            print("*" * 89 + "\n")

        all_val_losses.append(val_loss)
        
except KeyboardInterrupt:
    logging('-' * 100)
    logging('Exiting from training early')

# Load the best saved model.
with open(os.path.join(args.work_dir, 'model.pt'), 'rb') as f:
    model = torch.load(f)
para_model = model.to(device)

# Run on test data.
test_loss = evaluate(te_iter)
logging('=' * 100)
logging('| End of training | test loss {:5.2f} | test ppl {:9.3f}'.format(test_loss, math.exp(test_loss)))
logging('=' * 100)