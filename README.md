# Deep Equilibrium Models

**2020/5: The current version is not yet compatible with PyTorch 1.5 due to some issues with parameter replica in `DataParallel` (which has also affected some other transformer repos; see Issue [#3936](https://github.com/huggingface/transformers/issues/3936)). I will update the repo accordingly once this is fixed. For now, to run DEQ, PLEASE USE PyTorch <1.5.0.**

This repository contains the code for the deep equilibrium (DEQ) model, an implicit-depth architecture proposed in the paper [Deep Equilibrium Models](https://arxiv.org/abs/1909.01377) by Shaojie Bai, J. Zico Kolter and Vladlen Koltun.

Unlike many existing "deep" techniques, the DEQ model is a implicit-depth architecture that directly solves for and
backpropagates through the equilibrium state of an (effectively) infinitely deep network. Importantly, compared to 
prior implicit-depth approaches (e.g., ODE-based methods), in this work we also demonstrate the potential power and 
applicability of such models on practical, large-scale and high-dimensional sequence datasets. On these large-scale 
datasets, (carefully designed) DEQ models can acheive results on par with (or slightly better than) the SOTA 
deep networks, while not using a "deep" stacking (and with only O(1) memory). 

We provide two instantiations of DEQ here, based primarily on two SOTA sequence models: 1) universal transformers; 
and 2) trellis networks. More importantly, we have separated out a framework so that it requires minimal effort to 
try other interesting architectures/transformations beyond these two instantiations. See the README in `DEQModel/models` for more details. We also provide below URLs to the saved pre-trained models that achieve the state-of-the-art performance (e.g., 23.1 ppl on WT103).

If you find this repository useful for your research, please consider citing our work:
```
@inproceedings{bai2019deep,
  author    = {Shaojie Bai and J. Zico Kolter and Vladlen Koltun},
  title     = {Deep Equilibrium Models},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2019},
}
```

## News

2020/7: A branch `pytorch-1.5` has been created to resolve the DataParallel issue with PyTorch 1.5 (see [here](https://github.com/pytorch/pytorch/issues/40457) and [here](https://github.com/huggingface/transformers/pull/4300) for details). Specifically, this is not a DEQ-related issue, but one related to some of the modules DEQ depends on (e.g., adaptive embedding). In PyTorch 1.5, accessing parameters on the replicas is no longer possible. For now, you can use the `pytorch-1.5` to train the model from scratch, but there is no pre-trained model yet following the code change. To run pre-trained models, please still use PyTorch 1.4 and this `master` branch.

2020/2: Following the suggestions of many researchers, we have made a major update to the repo that significantly clarifies implementation structure of DEQ. Unlike the previous version (where `DEQFunc` and `DummyDEQFunc` could be confusing), both the forward and backward functionalities of DEQ are wrapped in the `DEQModule` class in file `module/deq.py`.

## Prerequisite

Python >= 3.5 and PyTorch >= 1.3.0. 4 GPUs strongly recommended for computational efficiency (although you could still fit in 1 GPU if needed).

## Data

You can download the dataset using 
```sh
bash get_data.sh
```

## Usage

All DEQ instantiations share the same underlying framework, whose core functionalities are provided in `DEQModel/modules`. In particular, `deq.py` provides the PyTorch functions that solves for the roots in forward and backward passes, **where the 
backward pass is hidden as an inner class of `DEQModule`**. `broyden.py` provides an implementation of the Broyden's method. Meanwhile, numerous regularization techniques (weight normalization, variational dropout, etc.) are provided in 
`optimizations.py` (heavily borrowed from the [TrellisNet](https://github.com/locuslab/trellisnet) repo).

Training and evaluation scripts of DEQ-Transformer and DEQ-TrellisNet are provided independently, in `DEQModel/train_[MODEL_NAME].py`. Most of the hyperparameters can be (and **should be**) tuned via the `argparse` flags.

#### Example Configuration Files
We also provide some sample scripts that run on 4-GPU machines (see `run_wt103_deq_[...].sh`). To execute these scripts, one can run (e.g. for a transformer with forward Broyden iteration limit set to 30):
```sh
bash run_wt103_deq_transformer.sh train --cuda --multi_gpu --f_thres 30 --b_thres 40 --subseq_len 75
```
**You should expect to get a test-set perplexity around 23.8 with this setting.**

The current repo contains the code/config files for the large-scale WikiText-103 language corpus. We will soon add the Penn TreeBank experiment and the copy memory task (which were also used in the paper).

#### File Structure

The files in this repo are organized in the following manner:

```
DEQModel/
  models/
    trellisnets/
      (2 files, one containing a TrellisNet LM and one its for-/backward DEQ operations)
    transformers/
      (2 files for similar purposes as above)
  modules/
     (Equilibrium solvers as well as regularization files)
  utils/
     (Extra language model related files, such as adaptive embeddings, etc.)
  LM-[...]deq-wt103/
     (All logs of the training, where [...] is the architecture type)
```

#### Pre-trained Models

We provide some reasonably good pre-trained weights here so that one can quickly play with DEQs without training from scratch.

| Description   | Task              | Dataset             | Model                                      | Expected Performance    |
| ------------- | ----------------- | ------------------- | ------------------------------------------ | ----------------------- |
| DEQ-Transformer | Word-Level Language Modeling | WikiText-103 | [download (.pkl)](https://drive.google.com/file/d/1I0q6f8-XFAEDqv-Zwi5Mxc9WtwmJT3sw/view?usp=sharing) |   23.1 Perplexity   |

To evaluate a trained model, simply use the `--load` flag and the `--eval` flag. Using the pretrained DEQ-Transformer on WT103 as an example (with the default parameters), with which you should expect to get a 23.1ppl (outperforming Transformer-XL's 23.6 ppl):

```
bash run_wt103_deq_transformer.sh train --f_thres 30 --eval --load [SAVED_MODEL_NAME].pkl --mem_len 300 --pretrain_step 0
```
(i.e., at inference time, set the augmented memory size to 300 (which you can adjust).)


## Tips

1. It is not easy to train a DEQ without knowing which hyperparameters you need to pay special attention to. Generally, the importance of these hyperparameters depend on the transformation f_\theta you choose for the architecture. For instance, each layer of the Transformer has a *much* larger receptive field than that of a TrellisNet, so we have observed that TrellisNet typically requires more Broyden steps to converge to the equilibrium (which means the Broyden iteration limit is typically larger).

2. Empirically, we find that training with subsequences makes the equilibrium solving process slightly more stable (especially when dealing with extremely long sequences). See the appendix in the paper for more details.

3. For most of the time, pre-training the model with a very shallow network (e.g., a 2-layer network) for a while (e.g., 10-20% of the total training steps/epochs) can be helpful, as it makes f_\theta more stable. However, note that these shallow networks themselves usually achieve very bad results on their own (e.g., imagine a 10-layer TrellisNet).

4. Patience. As the paper discusses, DEQ models could be slower than the corresponding "conventional" deep networks :P

5. Variational dropout typically makes equilibrium states harder to find. However, empirically, we find them to be extremely useful regularizations to these weight-tied models.

6. You can vary factors such as `--mem_len` (for DEQ-Transformer) and `--f_thres` at inference time. As we show in the paper, more Broyden steps typically yields (diminishingly) better results. Moreover, as DEQ only has "one layer", storage cost of the cached history sequence of size `--mem_len` is actually very cheap.


## Credits

The transformer implementation as well as the extra modules (e.g., adaptive embeddings) were based on the [Transformer-XL](https://github.com/kimiyoung/transformer-xl) repo.

The TrellisNet implementation as well as the weight-tying regularizations were based on the [TrellisNet](https://github.com/locuslab/trellisnet) repo.

We also added the RAdam optimizer as an option to the training (but didn't set it to default). The RAdam implementation is from the [RAdam](https://github.com/LiyuanLucasLiu/RAdam) repo.




