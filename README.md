# jax-vae
_Project has been extended to variational graph autoencoders in [salfaris/PriorVGAE](https://github.com/salfaris/PriorVGAE)._

Variational autoencoder implementation on categorical and continuous datasets using [dm-haiku](https://github.com/deepmind/dm-haiku).

The classification example is copied over from the haiku library. The aim is to classify hand-drawn digits to the correct number (a classic).

The continuous version is used for [encoding Gaussian process priors](https://arxiv.org/pdf/2110.10422.pdf), but can be extended to any continuous function.
