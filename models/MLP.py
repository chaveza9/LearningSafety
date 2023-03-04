from typing import Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp


def _matrix_init(key, shape, scale):
  """Initializes matrix with constant value according to init_scale.
  This initialization function was designed to avoid the early training
  stall due to too large or too small probability mass values computed
  with the initial parameters.
  Softplus is applied to the matrix during the forward pass, to ensure the
  matrix elements are all positive. With the current initialization, the
  matrix has constant values and after matrix-vector multiplication with
  the input, the output vector elements are constant
  ```
  x / init_scale ** (1 / 1 + len(self.features)).
  ```
  Assuming zero bias initialization and ignoring the factor residual
  paths, the CDF output of the entire distribution network is constant
  vector with value
  ```
  sigmoid(x / init_scale).
  ```
  Therefore `init_scale` should be in the order of the expected magnitude
  of `x`.
  Args:
    key: The Jax PRNG key for this matrix initialization.
    shape: Sequence of integers. The shape of the matrix.
    scale: Scale factor for initial value.
  Returns:
    The initial matrix value.
  """
  del key  # Unused.
  return jnp.full(shape, jnp.log(jnp.expm1(1 / scale / shape[-1])))


def _bias_init(key, shape):
  return jax.random.uniform(key, shape, minval=-.5, maxval=.5)


def _factor_init(key, shape):
  return jax.nn.initializers.zeros(key, shape)


class MonotonicMLP(nn.Module):
  """MLP that implements monotonically increasing functions by construction."""
  features: Tuple[int, ...]
  init_scale: float
  in_dim: int = 1
  out_dim: int = 1
  @nn.compact
  def __call__(self, x):
    scale = self.init_scale ** (1 / (1 + len(self.features)))
    u = x.reshape((-1, 1))
    features = (self.in_dim,) + self.features + (self.out_dim,)
    for k, shape in enumerate(zip(features[:-1], features[1:])):
      matrix = self.param(f"matrix_{k}", _matrix_init, shape, scale)
      bias = self.param(f"bias_{k}", _bias_init, shape[-1:])
      u = u @ jax.nn.softplus(matrix) + bias
      if k < len(self.features):
        factor = self.param(f"factor_{k}", _factor_init, shape[-1:])
        u += jnp.tanh(u) * jnp.tanh(factor)
    return u.reshape(x.shape)