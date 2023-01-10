# Safe Softmax

## Background

The softmax function of an input $x$ is defined as

$\textrm{softmax}(x_i) = \frac{exp(x_i)}{\sum_j exp(x_j)}$

PyTorch's [implementation][soft_py] implements this function and admits gradients. But there is an issue. If an entry in $x$ is `NaN` then the output of the `softmax` op is `NaN`.

## Not a solution!

One might suggest the following solution 

``` python
import torch
# construct an input tensor with a nan entry
x = torch.rand((10, 5))
x[5, 1] = float("nan")
x.requires_grad = True

# softmax and replace nan output with 0.0
y = torch.nn.functional.softmax(x, dim=1)
y = torch.nan_to_num(y, 0.0)
``` 

Why is this a problematic solution? Give an example.

## Question

Implement a `SafeSoftmax` operator that handles `NaN`s in the input elegantly and safely making outputs and gradients finite. 

The proposed op should look like this

```python
class _SafeSoftmax(torch.autograd.Function):
	@staticmethod
	def forward(ctx, x, dim=-1):
		<your code>

	@staticmethod
	def backward(ctx, grad):
		<your code>


safesoftmax = _SafeSoftmax.apply
```

Provide the necessary unit tests that guarantee the forward and backward correctness of your implementation.

[soft_py]: https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html