# Deep Equilibrium Models

This repository contains the code for the deep equilibrium (DEQ) model, an implicit-depth architecture proposed in the paper [Deep Equilibrium Models](https://arxiv.org/abs/1909.01377) by Shaojie Bai, J. Zico Kolter and Vladlen Koltun.

Unlike many existing "deep" techniques, the DEQ model is a implicit-depth architecture that directly solves for and
backpropagates through the equilibrium state of an (effectively) infinitely deep network. Importantly, compared to 
prior implicit-depth approaches (e.g., ODE-based methods), in this work we also demonstrate the potential power and 
applicability of such models on practical, large-scale and high-dimensional sequence datasets. On these large-scale 
datasets, (carefully designed) DEQ models can acheive results on par with (or even slightly better than) the SOTA 
deep networks, while not using a "deep" stacking (and with only O(1) memory). 

We provide two instantiations of DEQ here, based primarily on two SOTA sequence models: 1) universal transformers; 
and 2) trellis networks. But importantly, we have separated out a framework so that it requires minimal effort to 
try other interesting architectures/transformations. See the README in `DEQModel/models` for more details.

If you find this repository useful for your research, please consider citing our work:
```
@inproceedings{bai2019deep,
  author    = {Shaojie Bai and J. Zico Kolter and Vladlen Koltun},
  title     = {Deep Equilibrium Models},
  booktitle = {Neural Information Processing Systems (NeurIPS)},
  year      = {2019},
}
```

## Prerequisite

Python >= 3.5 and PyTorch >= 1.0.0. 4 GPUs strongly recommended for computational efficiency.

## Usage

All DEQ instantiations share the same underlying framework, whose core functionalities are provided in `DEQModel/modules`. In particular, `deq.py` provides the PyTorch functions that solves for the roots in forward and backward passes (you can also change the value of \epsilon there). `broyden.py` provides an implementation of the Broyden's method. Meanwhile, numerous regularization techniques (weight normalization, variational dropout, etc.) are provided in `optimizations.py` (heavily borrowed from the [TrellisNet](https://github.com/locuslab/trellisnet) repo).

Training and evaluation scripts of DEQ-Transformer and DEQ-TrellisNet are provided independently, in `DEQModel/train_[MODEL_NAME].py`. Most of the hyperparameters can be (and **should be**) tuned via the `argparse` flags. For instance:
```sh
python train_transformer.py --cuda --multi_gpu --d_embed 600 --d_model 600 --pretrain_steps 20000 [...]
```

#### Example Configuration Files
We also provide some sample scripts that runs on 4-GPU machines (see `run_wt103_deq_[...].sh`). To execute these scripts, one can run (e.g. for a transformer with forward Broyden iteration limit set to 30):
```sh
bash run_wt103_deq_transformer.sh train --cuda --multi_gpu --f_thres 30 --b_thres 50 --subseq_len 75
```
You should expect to get a 24 to 25 test-set perplexity with this setting.

The current repo contains the code/config files for the large-scale WikiText-103 language corpus. We will soon add the Penn TreeBank experiment and the copy memory task (which were also used in the paper).

#### File Structure

The files in this repo are organized in the following manner:

```
DEQModel/
  models/
    trellisnets/
      (2 files, one containing a TrellisNet LM and one containing its forward/backward DEQ operations)
    transformers/
      (2 files for similar purposes as above)
  modules/
     (Equilibrium solvers as well as regularization files)
  utils/
     (Extra language model related files, such as adaptive embeddings, etc.)
  LM-[...]deq-wt103/
     (All logs of the training, where [...] is the architecture type)
```

We will also release some pre-trained models in the near future.

## Tips

1. It is not easy to train a DEQ without knowing which hyperparameters you need to pay special attention to. Generally, the importance of these hyperparameters depend on the transformation f_\theta you choose for the architecture. For instance, each layer of the Transformer has a *much* larger receptive field than that of a TrellisNet, so we have observed that TrellisNet typically requires more Broyden steps to converge to the equilibrium (which means the Broyden iteration limit is typically larger).

2. Empirically, we find that training with subsequences makes the equilibrium solving process more stable (especially when dealing with extremely long sequences). See the appendix in the paper for more details.

3. For most of the time, pre-training the model with a very shallow network (e.g., a 2-layer network) for a while (e.g., 10-20% of the total training steps/epochs) can be helpful, as it makes f_\theta more stable. However, note that these shallow networks themselves usually achieve very bad results on their own (e.g., imagine a 10-layer TrellisNet).

4. Patience. As the paper discusses, DEQ models could be slower than the corresponding deep networks :P

(More to come)


## Credits

The transformer implementation as well as the extra modules (e.g., adaptive embeddings) were based on the [Transformer-XL](https://github.com/kimiyoung/transformer-xl) repo.

The TrellisNet implementation as well as the weight-tying regularizations were based on the [TrellisNet](https://github.com/locuslab/trellisnet) repo.

We also added the RAdam optimizer as an option to the training. The RAdam implementation is from the [RAdam](https://github.com/LiyuanLucasLiu/RAdam) repo.




