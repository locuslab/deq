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
from models.transformers.deq_transformer import DEQTransformerLM
from modules import radam
from utils.exp_utils import create_exp_dir
from utils.data_parallel import BalancedDataParallel


parser = argparse.ArgumentParser(description='PyTorch DEQ Sequence Model')
parser.add_argument('--data', type=str, default='../data/wikitext-103',
                    help='location of the data corpus (default to the WT103 path)')
parser.add_argument('--dataset', type=str, default='wt103',
                    choices=['wt103'],
                    help='dataset name')
parser.add_argument('--n_layer', type=int, default=12,
                    help='number of total layers')
parser.add_argument('--eval_n_layer', type=int, default=12,
                    help='number of total layers at evaluation')
parser.add_argument('--n_head', type=int, default=10,
                    help='number of heads (default: 10)')
parser.add_argument('--d_head', type=int, default=50,
                    help='head dimension (default: 50)')
parser.add_argument('--d_embed', type=int, default=-1,
                    help='embedding dimension (default: match d_model)')
parser.add_argument('--d_model', type=int, default=500,
                    help='model dimension (default: 500)')
parser.add_argument('--d_inner', type=int, default=8000,
                    help='inner dimension in the position-wise feedforward block (default: 8000)')

# Dropouts
parser.add_argument('--dropout', type=float, default=0.0,
                    help='global dropout rate (default: 0.05)')
parser.add_argument('--dropatt', type=float, default=0.0,
                    help='attention map dropout rate (default: 0.0)')

# Initializations
# Note: Generally, to make sure the DEQ model is stable initially, we should constrain the range
#       of initialization.
parser.add_argument('--init', default='normal', type=str,
                    help='parameter initializer to use.')
parser.add_argument('--emb_init', default='normal', type=str,
                    help='parameter initializer to use.')
parser.add_argument('--init_range', type=float, default=0.05,
                    help='parameters initialized by U(-init_range, init_range)')
parser.add_argument('--emb_init_range', type=float, default=0.01,
                    help='parameters initialized by U(-init_range, init_range)')
parser.add_argument('--init_std', type=float, default=0.01,
                    help='parameters initialized by N(0, init_std)')
parser.add_argument('--proj_init_std', type=float, default=0.01,
                    help='parameters initialized by N(0, init_std)')

# Optimizers
parser.add_argument('--optim', default='Adam', type=str,
                    choices=['Adam', 'SGD', 'Adagrad', 'RMSprop', 'RAdam'],
                    help='optimizer to use.')
parser.add_argument('--lr', type=float, default=0.00025,
                    help='initial learning rate (0.00025|5 for adam|sgd)')
parser.add_argument('--scheduler', default='cosine', type=str,
                    choices=['cosine', 'inv_sqrt', 'dev_perf', 'constant'],
                    help='lr scheduler to use.')
parser.add_argument('--warmup_step', type=int, default=0,
                    help='the number of steps to warm up the learning rate to its lr value')
parser.add_argument('--decay_rate', type=float, default=0.5,
                    help='decay factor when ReduceLROnPlateau is used')
parser.add_argument('--lr_min', type=float, default=0.0,
                    help='minimum learning rate during annealing')

# Gradient updates
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--clip_nonemb', action='store_true',
                    help='only clip the gradient of non-embedding params')
parser.add_argument('--max_step', type=int, default=200000,
                    help='upper epoch limit (at least 200K for WT103 or PTB)')
parser.add_argument('--batch_size', type=int, default=60,
                    help='batch size')
parser.add_argument('--batch_chunk', type=int, default=1,
                    help='split batch into chunks to save memory')

# Sequence logistics
parser.add_argument('--tgt_len', type=int, default=150,
                    help='number of tokens to predict')
parser.add_argument('--eval_tgt_len', type=int, default=150,
                    help='number of tokens to predict for evaluation')
parser.add_argument('--mem_len', type=int, default=150,
                    help='length of the retained previous heads')
parser.add_argument('--subseq_len', type=int, default=0,
                    help='length of subsequence processed each time by DEQ')
parser.add_argument('--local_size', type=int, default=0,
                    help='local horizon size')

# Training techniques
parser.add_argument('--not_tied', action='store_true',
                    help='do not tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--eval', action='store_true',
                    help='evaluation mode')
parser.add_argument('--adaptive', action='store_true',
                    help='use adaptive softmax')
parser.add_argument('--div_val', type=int, default=1,
                    help='divident value for adapative input and softmax')
parser.add_argument('--pre_lnorm', action='store_true',
                    help='apply LayerNorm to the input instead of the output')
parser.add_argument('--wnorm', action='store_true',
                    help='apply WeightNorm to the weights')
parser.add_argument('--varlen', action='store_true',
                    help='use variable length')
parser.add_argument('--multi_gpu', action='store_true',
                    help='use multiple GPU')
parser.add_argument('--log-interval', type=int, default=200,
                    help='report interval')
parser.add_argument('--eval-interval', type=int, default=4000,
                    help='evaluation interval')
parser.add_argument('--f_thres', type=int, default=50,
                    help='forward pass Broyden threshold')
parser.add_argument('--b_thres', type=int, default=80,
                    help='backward pass Broyden threshold')
parser.add_argument('--work_dir', default='LM-TFM', type=str,
                    help='experiment directory.')
parser.add_argument('--restart', action='store_true',
                    help='restart training from the saved checkpoint')
parser.add_argument('--restart_dir', type=str, default='',
                    help='restart dir')
parser.add_argument('--debug', action='store_true',
                    help='run in debug mode (do not create exp dir)')
parser.add_argument('--same_length', action='store_true',
                    help='use the same attn length for all tokens')
parser.add_argument('--attn_type', type=int, default=0,
                    help='attention type. 0 for ours, 1 for Shaw et al,'
                    '2 for Vaswani et al, 3 for Al Rfou et al. (Only 0 supported now)')
parser.add_argument('--eta_min', type=float, default=0.0,
                    help='min learning rate for cosine scheduler')
parser.add_argument('--weight_decay', type=float, default=0.0,
                    help='weight decay')
parser.add_argument('--gpu0_bsz', type=int, default=-1,
                    help='batch size on gpu 0')
parser.add_argument('--max_eval_steps', type=int, default=-1,
                    help='max eval steps')
parser.add_argument('--pretrain_steps', type=int, default=0,
                    help='number of pretrain steps')
parser.add_argument('--start_train_steps', type=int, default=0,
                    help='starting training step count (default to 0)')
parser.add_argument('--patience', type=int, default=0,
                    help='patience')
parser.add_argument('--load', type=str, default='',
                    help='path to load weight')
parser.add_argument('--name', type=str, default='N/A',
                    help='name of the trial')

args = parser.parse_args()
args.tied = not args.not_tied
args.pretrain_steps += args.start_train_steps
print(f"Experiment name: {args.name}")
assert args.mem_len > 0, "For now you must set mem_len > 0 when using deq"
args.work_dir += "deq"
args.cuda = torch.cuda.is_available()
    
if args.d_embed < 0:
    args.d_embed = args.d_model

assert args.batch_size % args.batch_chunk == 0

args.work_dir = '{}-{}'.format(args.work_dir, args.dataset)
args.work_dir = os.path.join(args.work_dir, time.strftime('%Y%m%d-%H%M%S'))
logging = create_exp_dir(args.work_dir,
    scripts_to_save=['train_transformer.py', 'models/transformers/deq_transformer.py'], debug=args.debug)

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
tr_iter = corpus.get_iterator('train', args.batch_size, args.tgt_len, device=device)
va_iter = corpus.get_iterator('valid', eval_batch_size, args.eval_tgt_len, device=device)
te_iter = corpus.get_iterator('test', eval_batch_size, args.eval_tgt_len, device=device)

# adaptive softmax / embedding
cutoffs, tie_projs = [], [False]
if args.adaptive:
    assert args.dataset in ['wt103']
    cutoffs = [20000, 40000, 200000]
    tie_projs += [True] * len(cutoffs)

###############################################################################
# Build the model
###############################################################################
def init_weight(weight):
    if args.init == 'uniform':
        nn.init.uniform_(weight, -args.init_range, args.init_range)
    elif args.init == 'normal':
        nn.init.normal_(weight, 0.0, args.init_std)

def init_bias(bias):
    nn.init.constant_(bias, 0.0)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1 or classname.find('Conv1d') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            init_weight(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            init_bias(m.bias)
    elif classname.find('AdaptiveEmbedding') != -1:
        if hasattr(m, 'emb_projs'):
            for i in range(len(m.emb_projs)):
                if m.emb_projs[i] is not None:
                    nn.init.normal_(m.emb_projs[i], 0.0, args.proj_init_std)
    elif classname.find('Embedding') != -1:
        if hasattr(m, 'weight'):
            init_weight(m.weight)
    elif classname.find('ProjectedAdaptiveLogSoftmax') != -1:
        if hasattr(m, 'cluster_weight') and m.cluster_weight is not None:
            init_weight(m.cluster_weight)
        if hasattr(m, 'cluster_bias') and m.cluster_bias is not None:
            init_bias(m.cluster_bias)
        if hasattr(m, 'out_projs'):
            for i in range(len(m.out_projs)):
                if m.out_projs[i] is not None:
                    nn.init.normal_(m.out_projs[i], 0.0, args.proj_init_std)
    elif classname.find('LayerNorm') != -1:
        if hasattr(m, 'weight'):
            nn.init.normal_(m.weight, 1.0, args.init_std)
        if hasattr(m, 'bias') and m.bias is not None:
            init_bias(m.bias)
    elif classname.find('WeightShareSelfAttention') != -1:
        if hasattr(m, 'r_w_bias'):
            init_weight(m.r_w_bias)
        if hasattr(m, 'r_r_bias'):
            init_weight(m.r_r_bias)

def update_dropout(m):
    classname = m.__class__.__name__
    if classname.find('Dropout') != -1:
        if hasattr(m, 'p'):
            m.p = args.dropout

def update_dropatt(m):
    if hasattr(m, 'dropatt'):
        m.dropatt.p = args.dropatt

if args.restart:
    with open(os.path.join(args.restart_dir, 'model.pt'), 'rb') as f:
        model = torch.load(f)
    model = model.float()
    model.apply(update_dropout)
    model.apply(update_dropatt)
else:
    model = DEQTransformerLM(ntokens, args.n_layer, args.eval_n_layer, args.n_head, args.d_model, args.d_head, args.d_inner,
                             args.dropout, args.dropatt, tie_weights=args.tied, d_embed=args.d_embed,
                             div_val=args.div_val, tie_projs=tie_projs, pre_lnorm=args.pre_lnorm,
                             wnorm=args.wnorm, local_size=args.local_size, pretrain_steps=args.pretrain_steps,
                             tgt_len=args.tgt_len, mem_len=args.mem_len, cutoffs=cutoffs, load=args.load)
    if len(args.load) == 0:
        model.apply(weights_init)    # Note: This applies weight_init recursively to modules in model
        model.word_emb.apply(weights_init)

args.n_all_param = sum([p.nelement() for p in model.parameters() if p.requires_grad])

if args.multi_gpu:
    model = model.to(device)
    if args.gpu0_bsz >= 0:
        para_model = BalancedDataParallel(args.gpu0_bsz // args.batch_chunk, model, dim=1).to(device)
    else:
        para_model = nn.DataParallel(model, dim=1).to(device)
else:
    para_model = model.to(device)

#### optimizer
optimizer = getattr(optim if args.optim != 'RAdam' else radam, args.optim)(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


#### scheduler
if args.scheduler == 'cosine':
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_step, eta_min=args.eta_min)
elif args.scheduler == 'inv_sqrt':
    # originally used for Transformer (in Attention is all you need)
    def lr_lambda(step):
        # return a multiplier instead of a learning rate
        if step == 0 and args.warmup_step == 0:
            return 1.
        else:
            return 1. / (step ** 0.5) if step > args.warmup_step else step / (args.warmup_step ** 1.5)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
elif args.scheduler == 'dev_perf':
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
        factor=args.decay_rate, patience=args.patience, min_lr=args.lr_min)

if args.restart:
    if os.path.exists(os.path.join(args.restart_dir, 'optimizer.pt')):
        with open(os.path.join(args.restart_dir, 'optimizer.pt'), 'rb') as f:
            opt_state_dict = torch.load(f)
            optimizer.load_state_dict(opt_state_dict)
    else:
        print('Optimizer was not saved. Start from scratch.')

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
    model.reset_length(args.eval_tgt_len, args.mem_len)

    # Evaluation
    total_len, total_loss = 0, 0.
    with torch.no_grad():
        mems = []
        for i, (data, target, seq_len) in enumerate(eval_iter):
            if 0 < args.max_eval_steps <= i:
                break
            ret = para_model(data, target, mems, train_step=train_step, f_thres=args.f_thres, 
                        b_thres=args.b_thres, subseq_len=subseq_len)
            loss, mems = ret[0], ret[1:]
            loss = loss.mean()
            total_loss += seq_len * loss.float().item()
            total_len += seq_len
    
    model.train()
    return total_loss / total_len


def train():
    global train_step, train_loss, best_val_loss, eval_start_time, log_start_time
    model.train()
    subseq_len = args.subseq_len
    model.reset_length(args.tgt_len, args.mem_len)

    train_iter = tr_iter.get_varlen_iter() if args.varlen else tr_iter

    if args.batch_chunk > 1:
        mems = [[] for _ in range(args.batch_chunk)]  # Each chunk (apparent) should have its own memory padding
    else:
        mems = []

    for batch, (data, target, seq_len) in enumerate(train_iter):
        model.zero_grad()
        if args.batch_chunk > 1:
            # Mode 1: Using accumulated gradient to train on a larger (effective) batch size
            data_chunks = data.chunk(args.batch_chunk, dim=1)
            target_chunks = target.chunk(args.batch_chunk, dim=1)
            for i in range(args.batch_chunk):
                data_i = data_chunks[i].contiguous()
                target_i = target_chunks[i].contiguous()
                ret = para_model(data_i, target_i, mems[i], train_step=train_step, f_thres=args.f_thres, 
                                 b_thres=args.b_thres, subseq_len=subseq_len)
                loss, mems[i] = ret[0], ret[1:]         # mems[i]: # 3 x bsz
                loss = loss.float().mean().type_as(loss) / args.batch_chunk
                loss.backward()
                train_loss += loss.float().item()
                
        else:
            # Mode 2: Normal training with one batch per iteration
            ret = para_model(data, target, mems, train_step=train_step, f_thres=args.f_thres, 
                             b_thres=args.b_thres, subseq_len=subseq_len)
            loss, mems = ret[0], ret[1:]
            loss = loss.float().mean().type_as(loss)
            loss.backward()
            train_loss += loss.float().item()
            
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        train_step += 1

        # Step-wise learning rate annealing according to some scheduling (we ignore 'constant' scheduling)
        if args.scheduler in ['cosine', 'constant', 'dev_perf']:
            # linear warmup stage
            if train_step < args.warmup_step:
                curr_lr = args.lr * train_step / args.warmup_step
                optimizer.param_groups[0]['lr'] = curr_lr
            else:
                if args.scheduler == 'cosine':
                    scheduler.step(train_step)
        elif args.scheduler == 'inv_sqrt':
            scheduler.step(train_step)

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

        # Enter evaluation/inference mode once in a while and save the model if needed
        if train_step % args.eval_interval == 0:
            val_loss = evaluate(va_iter)
            logging('-' * 100)
            log_str = '| Eval {:3d} at step {:>8d} | time: {:5.2f}s ' \
                      '| valid loss {:5.2f} | valid ppl {:9.3f}'.format(
                train_step // args.eval_interval, train_step,
                (time.time() - eval_start_time), val_loss, math.exp(val_loss))
            logging(log_str)
            logging('-' * 100)

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

            # dev-performance based learning rate annealing
            if args.scheduler == 'dev_perf':
                scheduler.step(val_loss)

            eval_start_time = time.time()
        
        if train_step == args.pretrain_steps and (args.pretrain_steps - args.start_train_steps) > 4000:
            print("You are using pre-training, which has completed :-)")
            model.save_weights(f"pretrain_{train_step}_{args.name}")
            torch.cuda.empty_cache()
            
        if train_step == args.max_step:
            break

# Loop over epochs.
train_step = args.start_train_steps
train_loss = 0
best_val_loss = None

log_start_time = time.time()
eval_start_time = time.time()

if args.eval:
    train_step = 1e9
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
        if train_step == args.max_step:
            logging('-' * 100)
            logging('End of training')
            break
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
