# Deep Equilibrium (DEQ) Sequence Models

This repository __mainly__ contains the code for the deep equilibrium transformer (DEQ-Transformer) model proposed in the paper [Deep Equilibrium Models](https://arxiv.org/abs/1909.01377) by Shaojie Bai, J. Zico Kolter and Vladlen Koltun.

We also provide additional support for regularizing the MDEQ models' stability, as introduced in the paper [Stabilizing Equilibrium Models by Jacobian Regularization](https://arxiv.org/abs/2106.14342).

If you find thie repository useful for your research, please consider citing our work:

```
@inproceedings{bai2019deep,
  author    = {Shaojie Bai and J. Zico Kolter and Vladlen Koltun},
  title     = {Deep Equilibrium Models},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2019},
}

@inproceedings{bai2021stabilizing,
  title     = {Stabilizing Equilibrium Models by Jacobian Regularization},
  author    = {Shaojie Bai and Vladlen Koltun and J. Zico Kolter},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2021}
}
```

### Requirements

PyTorch >=1.5.0, torchvision >= 0.4.0 recommended

### Dataset

You can download the dataset using 
```sh
bash get_data.sh
```

### Usage

For sequence modeling in particular, the default arguments to pass into the DEQ only include `z1s` (the hidden sequence we drive to equilibrium), 
`us` (input injection sequence), and `z0` (history padding to append to the left of `z1s`). Graphically:

```
  [<--------------- us --------------->]
  [              |                     ]         
  [<==== z0 ====>|<======= z1s =======>]
  [  (pad_len=L) |    (seq_len=L')     ]
(t=0)          (t=L)                (t=L+L')
```
In many cases, other arguments may be needed to compute the equilibrium (e.g., in transformers, positional encoding is critical). One can pass them in via the `*args` in the function statement.

##### 1. Train a DEQ-Transformer model on Wikitext-103 Dataset

You first need to download the dataset (see above). We also provide some sample scripts that run on 4-GPU machines (see `wt103_deq_[...].sh`). To execute these scripts, one can run (e.g. for a transformer with forward Broyden iteration limit set to 30):
```sh
bash wt103_deq_transformer.sh train --cuda --multi_gpu --f_solver broyden --f_thres 30 --b_thres 40
```
**You should expect to get a test-set perplexity around 23.8 with this setting.**

In this v2.0 DEQ repo, we now support both Broyden's method and Anderson acceleration methods for solving the fixed point, and one can choose different solvers for forward and backward processes. For example, to use Anderson in forward and Broyden in backward, one can simply do `--f_solver anderson --b_solver broyden`.

##### 2. Jacobian regularization
We also provide additional support for regularizing the stability of the MDEQ models. Specifically, we can do this efficiently by regularizing `||J_f||_F` at the equilibrium point (which characterizes fixed point models' stability) using the Hutchinson estimator. In practice, we can apply this regularization stochastically and adjust its strength dynamically. Please refer to the [Stabilizing Equilibrium Models by Jacobian Regularization](https://arxiv.org/abs/2106.14342) paper for more details.

The "regularized" version scripts are named `wt103_[...]_reg.sh`; and the pre-ln Transformer setting is named explicitly with `[...]_preln_reg.sh`. When training the model, the Jacobian regularization settings should be tuned and controlled entirely by the `argparse` options (see the main `README` in the upper level directory), such as `--jac_loss_weight`. 

### Pre-trained Models

We provide some reasonably good pre-trained weights here so that one can quickly play with DEQs without training from scratch.

| Description   | Task              | Dataset             | Model                                      | Expected Performance    |
| ------------- | ----------------- | ------------------- | ------------------------------------------ | ----------------------- |
| DEQ-Transformer | Word-Level Language Modeling | WikiText-103 | [download (.pkl)](https://drive.google.com/file/d/1lZx_sHt0-1gJVgXx90LDRizq3k-ZI0SW/view?usp=sharing) |   23.1 Perplexity   |
| DEQ-Transformer (reg.) | Word-Level Language Modeling | WikiText-103 | download (.pkl) |   23.8 Perplexity   |

To evaluate a pre-trained model, simply use the `--load` flag and the `--eval` flag. Using the pretrained DEQ-Transformer on WT103 as an example (with the default parameters), with which you should expect to get a 23.1ppl (outperforming Transformer-XL's 23.6 ppl):
```
bash run_wt103_deq_transformer.sh train --f_thres 30 --eval --load [SAVED_MODEL_NAME].pkl --mem_len 300 --pretrain_step 0
```
(i.e., at inference time, set the augmented memory size to 300 (which you can adjust to a different value).)


### Tips

1. For most of the time, pre-training the model with a very shallow network (e.g., a 2-layer network) for a while (e.g., 10-20% of the total training steps/epochs) can be helpful, as it makes f_\theta more stable. However, note that these shallow networks themselves usually achieve very bad results on their own.

2. Patience. As the paper discusses, DEQ models could be slower than the corresponding "conventional" deep networks :P

3. Variational dropout typically makes equilibrium states harder to find. However, empirically, we find them to be extremely useful regularizations to these weight-tied models.

4. You can vary factors such as `--mem_len` (for DEQ-Transformer) and `--f_thres` at inference time. As we show in the paper, more Broyden steps typically yields (diminishingly) better results. Moreover, as DEQ only has "one layer", storage cost of the cached history sequence of size `--mem_len` is actually very cheap.