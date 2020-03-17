#  Creating New Deep Equilibrium Models

> **Theorem 2 (Universality of "single-layer" DEQs, very informal)**: Stacking multiple DEQs 
> (with potentially _different_ classes of transformations) does not create extra representational
> power over a single DEQ.

(See the paper for a formal statement.)

By the theorem above, designing a better DEQ model boils down to designing a better stable transformation f_\theta. 
We provide two instantiations of DEQs in this directory:
  - TrellisNet (weight-tied temporal convolutions); and
  - Universal Transformer (weight-tied multi-head self-attention)

To create and play with new DEQs, we recommend following the same implementation structure, by creating the following 
two files: 
  1. `models/[MODEL_NAME]/deq_[MODEL_NAME].py` (which contains the definition of f_\theta); and 
  2. `models/[MODEL_NAME]/deq_[MODEL_NAME]_module.py` (which implements the `DEQModule` interface (see `deq.py`) that provides both forward equilibrium solving and backward implicit differentiation). Note that the backward pass is included as an inner class of the `DEQModule` class.
  
Specifically, when using the DEQ network (e.g., a DEQ-Transformer), all one needs to do are:
```py
func = TransformerLayer(...)                    # Initialize a layer f_\theta
func_copy = copy.deepcopy(func)                 # Make a copy of `func` (and turn off its `requires_grad`)
deq = TransformerDEQModule(func, func_copy)     # Instantiate a `DEQModule` object
...
output = deq(z, ...)                            # Call the DEQ module on input z
```

The default parameters to pass into the forward/backward methods only include `z1s` (hidden sequence), 
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
