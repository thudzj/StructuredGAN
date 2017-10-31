# StructuredGAN
This repo provides codes for "Structured Generative Adversarial Networks" and is based on [TripleGAN](https://github.com/zhenxuan00/triple-gan).

For example, you can run `python -u sgan_cifar10.py -ssl_seed 1` to reproduce the semi-supervised classification results on CIFAR-10 dataset.

You can run `python -u generate.py -oldmodel ...` to generate samples, infer latent codes and transfer image styles based on a trained model.
