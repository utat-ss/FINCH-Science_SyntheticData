# FINCH-Science_SyntheticData
Code to synthetically generate new data, for the use of further training of hyperspectral unmixing algorithms.

# Description

This repository is used for the development of synthetic data to be used by Science. Science needs to generate synthetic data (to be used in atmospheric modelling) in order to increase "the space" at which we are training our unmixing algorithms, giving us a better training set.

# Models 

To perform this, there are a couple of models being explored currently. These are as follows:

- Conditional Convolutional Variational Auto Encoder (CCVAE): This model is used because of its capability to being conditioned on the abundances of EMs in spectra and ease of training.
- Diffusion (Diff): Diffusion models will be explored given their extensive capability to learn while taking longer to train.
- Generative Adversarial Network (GAN): GAN models will be explored given their ease of training.