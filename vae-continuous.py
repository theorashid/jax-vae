"""Variational Autoencoder example on continuous Gaussian process priors."""

from typing import Iterator, NamedTuple, Sequence, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
from absl import app, flags, logging
from tinygp import GaussianProcess, kernels

flags.DEFINE_integer("train_size", 16000, "Size of the training dataset.")
flags.DEFINE_integer("test_size", 4000, "Size of the testing dataset.")
flags.DEFINE_integer("batch_size", 128, "Size of the batch to train on.")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate for the optimizer.")
flags.DEFINE_integer("training_steps", 10000, "Number of training steps.")
flags.DEFINE_integer("eval_frequency", 100, "How often to evaluate the model.")
flags.DEFINE_integer("random_seed", 42, "Random seed.")
FLAGS = flags.FLAGS


PRNGKey = jnp.ndarray
Batch = jnp.ndarray

SAMPLE_SHAPE: Sequence[int] = (100, 1)


def generate_gp_samples(
    X: jnp.ndarray,
    var: float,
    scale: float,
    num_draws: int,
    batch_size: int,
    sample_shape: Sequence[int] = SAMPLE_SHAPE,
    seed: int = 1,
) -> Iterator[Batch]:
    kernel = var * kernels.ExpSquared(scale=scale)
    gp = GaussianProcess(kernel, X)

    draws = gp.sample(
        jax.random.PRNGKey(seed=seed),
        shape=(
            num_draws,
            batch_size,
        ),
    )

    draws = jnp.reshape(draws, (-1, *(batch_size, *sample_shape)))

    return iter(draws)


class Encoder(hk.Module):
    """Encoder model."""

    def __init__(
        self,
        hidden_size1: int = 50,
        hidden_size2: int = 25,
        latent_size: int = 10,
    ):
        super().__init__()
        self._hidden_size1 = hidden_size1
        self._hidden_size2 = hidden_size2
        self._latent_size = latent_size
        self.act = jax.nn.relu

    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x = hk.Flatten()(x)
        x = hk.Sequential(
            [
                hk.Linear(self._hidden_size1),
                self.act,
                hk.Linear(self._hidden_size2),
                self.act,
            ]
        )(x)

        mean = hk.Linear(self._latent_size)(x)
        log_stddev = hk.Linear(self._latent_size)(x)
        stddev = jnp.exp(log_stddev)

        return mean, stddev


class Decoder(hk.Module):
    """Decoder model."""

    def __init__(
        self,
        hidden_size1: int = 25,
        hidden_size2: int = 50,
        output_shape: Sequence[int] = SAMPLE_SHAPE,
    ):
        super().__init__()
        self._hidden_size1 = hidden_size1
        self._hidden_size2 = hidden_size2
        self._output_shape = output_shape
        self.act = jax.nn.relu

    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        output = hk.Sequential(
            [
                hk.Linear(self._hidden_size1),
                self.act,
                hk.Linear(self._hidden_size2),
                self.act,
                hk.Linear(np.prod(self._output_shape)),
            ]
        )(z)

        output = jnp.reshape(output, (-1, *self._output_shape))

        return output


class VAEOutput(NamedTuple):
    mean: jnp.ndarray
    stddev: jnp.ndarray
    output: jnp.ndarray


class VariationalAutoEncoder(hk.Module):
    """Main VAE model class, uses Encoder & Decoder under the hood."""

    def __init__(
        self,
        hidden_size1: int = 50,
        hidden_size2: int = 25,
        latent_size: int = 10,
        output_shape: Sequence[int] = SAMPLE_SHAPE,
    ):
        super().__init__()
        self._hidden_size1 = hidden_size1
        self._hidden_size2 = hidden_size2
        self._latent_size = latent_size
        self._output_shape = output_shape

    def __call__(self, x: jnp.ndarray) -> VAEOutput:
        x = x.astype(jnp.float32)
        mean, stddev = Encoder(
            self._hidden_size1, self._hidden_size2, self._latent_size
        )(x)
        z = mean + stddev * jax.random.normal(hk.next_rng_key(), mean.shape)
        output = Decoder(
            self._hidden_size2,
            self._hidden_size1,
            self._output_shape,
        )(z)

        return VAEOutput(mean, stddev, output)


def mean_squared_error(x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
    """Calculate mean squared error between two tensors.

    Args:
            x1: variable tensor
            x2: variable tensor, must be of same shape as x1

    Returns:
            A scalar representing mean square error for the two input tensors.
    """
    if x1.shape != x2.shape:
        raise ValueError("x1 and x2 must be of the same shape")

    x1 = jnp.reshape(x1, (x1.shape[0], -1))
    x2 = jnp.reshape(x2, (x2.shape[0], -1))

    return jnp.mean(jnp.square(x1 - x2), axis=-1)


def kl_gaussian(mean: jnp.ndarray, var: jnp.ndarray) -> jnp.ndarray:
    r"""Calculate KL divergence between given and standard gaussian distributions.

    KL(p, q) = H(p, q) - H(p) = -\int p(x)log(q(x))dx - -\int p(x)log(p(x))dx
            = 0.5 * [log(|s2|/|s1|) - 1 + tr(s1/s2) + (m1-m2)^2/s2]
            = 0.5 * [-log(|s1|) - 1 + tr(s1) + m1^2] (if m2 = 0, s2 = 1)

    Args:
        mean: mean vector of the first distribution
        var: diagonal vector of covariance matrix of the first distribution

    Returns:
        A scalar representing KL divergence of the two Gaussian distributions.
    """
    return 0.5 * jnp.sum(-jnp.log(var) - 1.0 + var + jnp.square(mean), axis=-1)


def main(_):
    FLAGS.alsologtostderr = True

    model = hk.transform(
        lambda x: VariationalAutoEncoder()(x)
    )  # pylint: disable=unnecessary-lambda
    optimizer = optax.adam(FLAGS.learning_rate)

    @jax.jit
    def loss_fn(
        params: hk.Params,
        rng_key: PRNGKey,
        batch: Batch,
    ) -> jnp.ndarray:
        """ELBO: E_p[log(x)] - KL(d||q), where p ~ Be(0.5) and q ~ N(0,1)."""
        outputs: VAEOutput = model.apply(params, rng_key, batch)

        log_likelihood = -mean_squared_error(batch, outputs.output)
        kl = kl_gaussian(outputs.mean, jnp.square(outputs.stddev))
        elbo = log_likelihood - kl

        return -jnp.mean(elbo)

    @jax.jit
    def update(
        params: hk.Params,
        rng_key: PRNGKey,
        opt_state: optax.OptState,
        batch: Batch,
    ) -> Tuple[hk.Params, optax.OptState]:
        """Single SGD update step."""
        grads = jax.grad(loss_fn)(params, rng_key, batch)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state

    rng_seq = hk.PRNGSequence(FLAGS.random_seed)
    params = model.init(next(rng_seq), np.zeros((1, *SAMPLE_SHAPE)))
    opt_state = optimizer.init(params)

    X = jnp.linspace(0, 10, 100)

    train_ds = generate_gp_samples(
        X,
        var=1.0,
        scale=1.0,
        num_draws=FLAGS.train_size,
        batch_size=FLAGS.batch_size,
    )
    valid_ds = generate_gp_samples(
        X,
        var=1.0,
        scale=1.0,
        num_draws=FLAGS.test_size,
        batch_size=FLAGS.batch_size,
    )

    for step in range(FLAGS.training_steps):
        params, opt_state = update(
            params,
            next(rng_seq),
            opt_state,
            next(train_ds),
        )

        if step % FLAGS.eval_frequency == 0:
            val_loss = loss_fn(params, next(rng_seq), next(valid_ds))
            logging.info("STEP: %5d; Validation ELBO: %.3f", step, -val_loss)


if __name__ == "__main__":
    app.run(main)
