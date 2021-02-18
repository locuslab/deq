#  Creating New Deep Equilibrium Models

(v2.0 of the implementation; updated in 2/2021)

> **Theorem 2 (Universality of "single-layer" DEQs, very informal)**: Stacking multiple DEQs 
> (with potentially _different_ classes of transformations) does not create extra representational
> power over a single DEQ.

(See the paper for a formal statement.)

By the theorem above, designing a better DEQ model boils down to designing a better stable transformation f_\theta. We provide an instantiation of DEQ in this directory, via *weight-tied multi-head self-attention*.

Creating and playing with a DEQ is **easy**. In general, we break our recommended implementation of a generic DEQ into 3 parts:

### Step 1. Defining a layer `f`=f_\theta that we'd like to iterate until equilibrium.

Typically, this is just like any deep network layer, and should be a subclass of `torch.nn.Module`. Evaluating this layer requires the hidden unit `z` and the input injection `x`; e.g.:
```python
class Layer(nn.Module):
    def __init__(self, ...):
	...
    def forward(self, z, x, **kwargs):
        return new_z
```
In `deq_transformer.py`, we provide an example of this with the class `RelPartialLearnableDecoderLayer`.

### Step 2. Prepare the fixed point solver to use for the DEQ model.

As a DEQ model can use any *black-box* root solver, we can implement and use any solver as long as it gives us a good estimate of the fixed point. In `../modules/solvers.py`, we provide two popular & efficient fixed point solvers, which are based on [Anderson acceleration](https://en.wikipedia.org/wiki/Anderson_acceleration) and [Broyden's method](https://en.wikipedia.org/wiki/Broyden%27s_method), respectively. These two methods (`anderson(...)` and `broyden(...)`) outputs a dictionary that contains the basic information of the optimization process. By default, we use the *relative residual difference* (i.e., |f(z)-z|/|z|) as the criterion for stopping the iterative process.

In the DEQ network, the forward pass can then be reduced to 2 lines:
```python
with torch.no_grad():
    # x is the input injection; z0 is the initial estimate of the fixed point.
    z_star = self.solver(lambda z: f(z, x, *args), z0, threshold=f_thres)['result']
```
where we note that the forward pass does not need to store **any** intermediate state, so we put it in a `torch.no_grad()` block. Also, note that we directly pass in the layer `f` defined in step 1, rather than the residual function `f(z)-z`.

### Step 3. Engage with the autodiff tape in order to use implicit differentiation

Finally, we need to ensure there is a way to compute the backward pass of a DEQ, which relies on implicit function theorem. To do this, we can use the `register_hook` function in PyTorch that registers a backward hook function to be executed in the backward pass. As we noted in the paper, the backward pass is simply solving for the fixed point of a *linear system* involving the Jacobian at the equilibrium. 

A full DEQ model implementation is therefore as simple as follows:
```python
from solvers import anderson, broyden

class DEQModel(nn.Module):
    def __init__(self, ...):
        super().__init__(...)
        self.f = Layer(...)   # See step 1 above
        self.solver = broyden
    
    def forward(self, x, ..., **kwargs):
        z0 = torch.zeros(...)

        # Forward pass
        with torch.no_grad():
            z_star = self.solver(lambda z: self.f(z, x, *args), z0, threshold=f_thres)['result']   # See step 2 above
            new_z_star = z_star

        # (Prepare for) Backward pass
        if self.training:
            z_star.requires_grad()
            new_z_star = self.f(z_star, x, *args)

            # Have to use a copy here because a hook will be applied to new_z_star (which could otherwise 
            # cause infinite recursion)
            z_star_copy = z_star.clone().detach().requires_grad_()
            new_z_star_copy =  self.f(z_star_copy, x, *args)
            def backward_hook(grad):
                # Compute the fixed point of yJ + grad, where J=J_f is the Jacobian of f at z_star
                new_grad = self.solver(lambda y: autograd.grad(new_z_star_copy, z_star_copy, y, retain_graph=True)[0] + grad, \
                                       torch.zeros_like(grad), threshold=b_thres)['result']
                return new_grad

            new_z_star.register_hook(backward_hook)
        return new_z_star
```

### The DEQ-Transformer Instantiation

For sequence modeling in particular, the default arguments to pass into the DEQ only include `z1s` (the hidden sequence we drive to equilibrium), 
`us` (input injection sequence), and `z0` (history padding to append to the left of `z1s`). Graphically:

```
  [<--------------- us --------------->]
  [              |                     ]         
  [<==== z0 ====>|<======= z1s =======>]
  [  (pad_len=L) |    (seq_len=L')     ]
(t=0)          (t=L)                (t=L+L')
```
In many cases, other arguments may be needed to compute the equilibrium (e.g., in transformers, positional encoding
is critical). One can pass them in via the `*args` in the function statement.
