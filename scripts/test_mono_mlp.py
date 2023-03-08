from typing import Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core import freeze, unfreeze
from jax import random                 # JAX random number generator
import optax                           # Common loss functions and optimizers
import matplotlib.pyplot as plt

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

  @nn.compact
  def __call__(self, x):
    scale = self.init_scale ** (1 / (1 + len(self.features)))
    u = x.reshape((-1, 1))
    # u = x
    features = (1,) + self.features + (1,)
    for k, shape in enumerate(zip(features[:-1], features[1:])):
      matrix = self.param(f"matrix_{k}", _matrix_init, shape, scale)
      bias = self.param(f"bias_{k}", _bias_init, shape[-1:])
      u = u @ jax.nn.softplus(matrix) + bias
      if k < len(self.features):
        factor = self.param(f"factor_{k}", _factor_init, shape[-1:])
        u += jnp.tanh(u) * jnp.tanh(factor)
    return u.reshape(x.shape)
    # return u

@jax.jit
def f(x_1):
    return .001*(x_1**3 + x_1) + jnp.tanh(x_1)

key1, key2 = jax.random.split(jax.random.PRNGKey(0))
def create_dataset(n_samples, key):
    x = jax.random.normal(key, (n_samples, 1))
    y = f(x)
    return x, y

model = MonotonicMLP(features=(10, 10, 10), init_scale=1.)
x = jax.random.normal(key1, (10,1)) # Dummy input data
params = model.init(key2, x) # Initialization call
jax.tree_util.tree_map(lambda x: x.shape, params) # Checking output shapes

# Set problem dimensions.
n_samples = 10000
x_dim = 10
y_dim = 5


# Generate samples with additional noise.
key_sample, key_noise = random.split(key1)

x_samples, y_samples = create_dataset(n_samples, key_sample)

print('x shape:', x_samples.shape, '; y shape:', y_samples.shape)

@jax.jit
def mse(params, x_batched, y_batched):
  # Define the squared loss for a single pair (x,y)
  def squared_error(x, y):
    pred = model.apply(params, x)
    return jnp.inner(y-pred, y-pred) / 2.0
  # Vectorize the previous to compute the average of the loss on all samples.
  return jnp.mean(jax.vmap(squared_error)(x_batched,y_batched), axis=0)

learning_rate = 0.01  # Gradient step size.

tx = optax.adam(learning_rate=learning_rate)
opt_state = tx.init(params)
loss_grad_fn = jax.value_and_grad(mse)

for i in range(20000):
  loss_val, grads = loss_grad_fn(params, x_samples, y_samples)
  updates, opt_state = tx.update(grads, opt_state)
  params = optax.apply_updates(params, updates)
  if i % 10 == 0:
    print(i)
    print('Loss step {}: '.format(i), loss_val)
# <<TEST>>
x = jnp.expand_dims(jnp.arange(-5, 5, .1),1)
y = f(x[:,0])
y_hat = model.apply(params, x)

plt.plot(x, y_hat, label="Monotonic model")
plt.plot(x, y, label="groundtruth")
plt.legend()
plt.grid()
plt.show()
plt.savefig("Monotonicity.png")