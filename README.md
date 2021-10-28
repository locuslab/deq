# Deep Equilibrium Models

> (Version 2.0 released now! :grinning:)

## News

:boom:**2021/6: Repo updated with the multiscale DEQ (MDEQ) code, Jacobian-related analysis & regularization support, and the new, faster and simpler implicit differentiation implementation through PyTorch's backward hook! (See [here](https://github.com/locuslab/deq#how-to-buildtrain-a-deq-model).)**

- For those who would like to start with a toy version of the DEQ, the NeurIPS 2020 tutorial on "Deep Implicit Layers" has a detailed step-by-step introduction: [tutorial video & colab notebooks here](http://implicit-layers-tutorial.org/).

- A [JAX](https://github.com/google/jax) version of the DEQ, including JAX implementation of Broyden's method, etc. is available [here](https://github.com/akbir/deq-jax).

---

This repository contains the code for the deep equilibrium (DEQ) model, an implicit-depth architecture that directly solves for and backpropagtes through the (fixed-point) equilibrium state of an (effectively) infinitely deep network. Importantly, compared to prior implicit-depth approaches (e.g., ODE-based methods), in this work we also demonstrate the potential power and compatibility of this implicit model with modern, structured layers like Transformers, which enable the DEQ networks to achieve results on par with the SOTA deep networks (in NLP and vision) *without* using a "deep" stacking (and thus O(1) memory). Moreover, we also provide tools for regularizing the stability of these implicit models.

Specifically, this repo contains the code from the following papers (see `bibtex` at the end of this README):
  - [Deep Equilibrium Models](https://arxiv.org/abs/1909.01377)
  - [Multiscale Deep Equilibrium Models](https://arxiv.org/abs/2006.08656)
  - [Stabilizing Equilibrium Models by Jacobian Regularization](https://arxiv.org/abs/2106.14342).

## Prerequisite

Python >= 3.6 and PyTorch >= 1.10. 4 GPUs strongly recommended for computational efficiency.

## Data

We provide more detailed instructions for downloading/processing the datasets (WikiText-103, ImageNet, Cityscapes, etc.) in the `DEQ-Sequence/` and `MDEQ-Vision/` subfolders.

## How to build/train a DEQ model?

Starting in 2021/6, we partition the repo into two sections, containing the sequence-model DEQ (i.e., `DEQ-Sequence/`) and the vision-model DEQ (i.e., `MDEQ-Vision/`) networks, respectively. As these two tasks require different input processing and loss objectives, they do not directly share the training framework. 

However, both frameworks share the same utility code, such as:
  - `lib/solvers.py`: Advanced fixed-point solvers (e.g., Anderson acceleration and Broyden's method)
  - `lib/jacobian.py`: Jacobian-related estimations (e.g., Hutchinson estimator and the Power method)
  - `lib/optimization.py`: Regularizations (e.g., weight normalization and variational dropout)
  - `lib/layer_utils.py`: Layer utilities

Moreover, the repo is significantly simplified from the previous version for users to extend on it. In particular, 

>**Theorem 2 (Universality of "single-layer" DEQs, very informal)**: Stacking multiple DEQs 
> (with potentially _different_ classes of transformations) does not create extra representational
> power over a single DEQ.

(See the paper for a formal statement.) By the theorem above, designing a better DEQ model boils down to designing a better stable transformation f_\theta. Creating and playing with a DEQ is **easy**, and we recommend following 3 steps (which we adopt in this repo):

### Step 1: Defining a layer `f=f_\theta` that we'd like to iterate until equilibrium.

Typically, this is just like any deep network layer, and should be a subclass of `torch.nn.Module`. Evaluating this layer requires the hidden unit `z` and the input injection `x`; e.g.:
```python
class Layer(nn.Module):
    def __init__(self, ...):
	...
    def forward(self, z, x, **kwargs):
        return new_z
```

### Step 2: Prepare the fixed point solver to use for the DEQ model.

As a DEQ model can use any *black-box* root solver. We provide PyTorch fixed-point solver implementations `anderson(...)` and `broyden(...)` in `lib/solvers.py` that output a dictionary containing the basic information of the optimization process. By default, we use the *relative residual difference* (i.e., |f(z)-z|/|z|) as the criterion for stopping the iterative process.

The forward pass can then be reduced to 2 lines:
```python
with torch.no_grad():
    # x is the input injection; z0 is the initial estimate of the fixed point.
    z_star = self.solver(lambda z: f(z, x, *args), z0, threshold=f_thres)['result']
```
where we note that the forward pass does not need to store **any** intermediate state, so we put it in a `torch.no_grad()` block.

### Step 3: Engage with the autodiff tape to use implicit differentiation

Finally, we need to ensure there is a way to compute the backward pass of a DEQ, which relies on implicit function theorem. To do this, we can use the `register_hook` function in PyTorch that registers a backward hook function to be executed in the backward pass. As we noted in the paper, the backward pass is simply solving for the fixed point of a *linear system* involving the Jacobian at the equilibrium:
```python
new_z_star = self.f(z_star.requires_grad_(), x, *args)

def backward_hook(grad):
    if self.hook is not None:
        self.hook.remove()
        torch.cuda.synchronize()   # To avoid infinite recursion
    # Compute the fixed point of yJ + grad, where J=J_f is the Jacobian of f at z_star
    new_grad = self.solver(lambda y: autograd.grad(new_z_star, z_star, y, retain_graph=True)[0] + grad, \
                           torch.zeros_like(grad), threshold=b_thres)['result']
    return new_grad

self.hook = new_z_star.register_hook(backward_hook)
```

### (Optional) Additional Step: Jacobian Regularization.

The fixed-point formulation of DEQ models means their stability are directly characterized by the Jacobian matrix `J_f` at the equilibrium point. Therefore, we provide code for analyzing and regularizing the Jacobian properties (based on the ICML'21 paper [Stabilizing Equilibrium Models by Jacobian Regularization](https://arxiv.org/abs/2106.14342)). Specifically, we added the following flags to the training script:

  - `jac_loss_weight`: The strength of Jacobian regularization, where we regularize `||J_f||_F`.
  - `jac_loss_freq`: The frequency `p` of the stochastic Jacobian regularization (i.e., we only apply this loss with probaility `p` during training).
  - `jac_incremental`: If >0, then we increase the `jac_loss_weight` by 0.1 after every `jac_incremental` training steps.
  - `spectral_radius_mode`: If `True`, estimate the DEQ models' spectral radius when evaluating on the validation set.

A full DEQ model implementation is therefore as simple as follows:
```python
from lib.solvers import anderson, broyden
from lib.jacobian import jac_loss_estimate

class DEQModel(nn.Module):
    def __init__(self, ...):
        ...
        self.f = Layer(...)
        self.solver = broyden
        ...
    
    def forward(self, x, ..., **kwargs):
        z0 = torch.zeros(...)

        # Forward pass
        with torch.no_grad():
            z_star = self.solver(lambda z: self.f(z, x, *args), z0, threshold=f_thres)['result']   # See step 2 above
            new_z_star = z_star

        # (Prepare for) Backward pass, see step 3 above
        if self.training:
            new_z_star = self.f(z_star.requires_grad_(), x, *args)
            
            # Jacobian-related computations, see additional step above. For instance:
            jac_loss = jac_loss_estimate(new_z_star, z_star, vecs=1)

            def backward_hook(grad):
                if self.hook is not None:
                    self.hook.remove()
                    torch.cuda.synchronize()   # To avoid infinite recursion
                # Compute the fixed point of yJ + grad, where J=J_f is the Jacobian of f at z_star
                new_grad = self.solver(lambda y: autograd.grad(new_z_star, z_star, y, retain_graph=True)[0] + grad, \
                                       torch.zeros_like(grad), threshold=b_thres)['result']
                return new_grad

            self.hook = new_z_star.register_hook(backward_hook)
        return new_z_star, ...
```

## Fixed-point Solvers

We provide PyTorch implementation of two generic solvers, `broyden(...)` (based on Broyden's method) and `anderson(...)` (based on Anderson acceleration) in `lib/solvers.py`. Both functions take in the transformation `f` whose fixed point we would like to solve for, and returns a dictionary of the following format:
```
{
 "result": ... (The closest estimate to the fixed point),
 "nstep": ... (The step that gives us this closest estimate),
 "abs_trace": ... (Absolute residuals along the trajectory),
 "rel_trace": ... (Relative residuals along the trajectory),
 ...
}
```

## Pretrained Models

See `DEQ-Sequence/` and `MDEQ-Vision/` sub-directories for the links.

## Credits

- The transformer implementation as well as the extra modules (e.g., adaptive embeddings) were based on the [Transformer-XL](https://github.com/kimiyoung/transformer-xl) repo.

- Some utilization code (e.g., model summary and yaml processing) of this repo were modified from the [HRNet](https://github.com/HRNet/HRNet-Semantic-Segmentation) repo.

- We also added the RAdam optimizer as an option to the training (but didn't set it to default). The RAdam implementation is from the [RAdam](https://github.com/LiyuanLucasLiu/RAdam) repo.

## Bibtex

If you find this repository useful for your research, please consider citing our work(s):

1. Deep Equilibrium Models
```
@inproceedings{bai2019deep,
  author    = {Shaojie Bai and J. Zico Kolter and Vladlen Koltun},
  title     = {Deep Equilibrium Models},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2019},
}
```

2. Multiscale Deep Equilibrium Models
```
@inproceedings{bai2020multiscale,
  author    = {Shaojie Bai and Vladlen Koltun and J. Zico Kolter},
  title     = {Multiscale Deep Equilibrium Models},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2020},
}
```

3. Stabilizing Equilibrium Models by Jacobian Regularization
```
@inproceedings{bai2021stabilizing,
  title     = {Stabilizing Equilibrium Models by Jacobian Regularization},
  author    = {Shaojie Bai and Vladlen Koltun and J. Zico Kolter},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2021}
}
```


