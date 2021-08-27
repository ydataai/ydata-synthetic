![](https://img.shields.io/github/workflow/status/ydataai/ydata-synthetic/prerelease)
![](https://img.shields.io/pypi/status/ydata-synthetic)
[![](https://pepy.tech/badge/ydata-synthetic)](https://pypi.org/project/ydata-synthetic/)
![](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8-blue)
[![](https://img.shields.io/pypi/v/ydata-synthetic)](https://pypi.org/project/ydata-synthetic/)
![](https://img.shields.io/github/license/ydataai/ydata-synthetic)

<p align="center"><img width="200" src="https://ydata-demos.s3.eu-central-1.amazonaws.com/Synthetic+Data_2.png" alt="Synthetic Data Logo"></p>

Join us on [![slack](https://img.shields.io/badge/slack-brightgreen.svg?logo=slack)](http://slack.ydata.ai/)

# What is Synthetic Data?
Synthetic data is artificially generated data that is not collected from real world events. It replicates the statistical components of real data without containing any identifiable information, ensuring individuals' privacy.

# Why Synthetic Data?
Synthetic data can be used for many applications:
- Privacy
- Remove bias
- Balance datasets
- Augment datasets

# ydata-synthetic
This repository contains material related with Generative Adversarial Networks for synthetic data generation, in particular regular tabular data and time-series. 
It consists a set of different GANs architectures developed using Tensorflow 2.0. Several example Jupyter Notebooks and Python scripts are included, to show how to use the different architectures.

# Quickstart

The source code is currently hosted on GitHub at: https://github.com/ydataai/ydata-synthetic

Binary installers for the latest released version are available at the [Python Package Index (PyPI).](https://pypi.org/project/ydata-synthetic/)
```
pip install ydata-synthetic
```

## Examples
Here you can find usage examples of the package and models to synthesize tabular data.

- Synthesizing the minority class with VanillaGAN on credit fraud dataset  [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ydataai/ydata-synthetic/blob/master/examples/regular/gan_example.ipynb)
- Time Series synthetic data generation with TimeGAN on stock dataset [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ydataai/ydata-synthetic/blob/master/examples/timeseries/TimeGAN_Synthetic_stock_data.ipynb)
- More examples are continously added and can be found in `/examples` directory.

# Project Resources

In this repository you can find the several GAN architectures that are used to create synthesizers:

#### Tabular data
- [GAN](https://arxiv.org/abs/1406.2661)
- [CGAN (Conditional GAN)](https://arxiv.org/abs/1411.1784)
- [WGAN (Wasserstein GAN)](https://arxiv.org/abs/1701.07875)
- [WGAN-GP (Wassertein GAN with Gradient Penalty)](https://arxiv.org/abs/1704.00028)
- [DRAGAN (On Convergence and stability of GANS)](https://arxiv.org/pdf/1705.07215.pdf)
- [Cramer GAN (The Cramer Distance as a Solution to Biased Wasserstein Gradients)](https://arxiv.org/abs/1705.10743)

#### Sequential data
- [TimeGAN](https://papers.nips.cc/paper/2019/file/c9efe5f26cd17ba6216bbe2a7d26d490-Paper.pdf)

# Contributing
We are open to collaboration! If you want to start contributing you only need to:
1. Search for an issue in which you would like to work. Issues for newcomers are labeled with good first issue.
2. Create a PR solving the issue.
3. We would review every PRs and either accept or ask for revisions.

# Support
For support in using this library, please join the #help Slack channel. The Slack community is very friendly and great about quickly answering questions about the use and development of the library. [Click here to join our Slack community!](http://slack.ydata.ai/)

# License
[GNU General Public License v3.0](https://github.com/ydataai/ydata-synthetic/blob/master/LICENSE)
