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
It consists in a set of different GANs architectures developed ussing Tensorflow 2.0. An example Jupyter Notebook is included, to show how to use the different architectures.

# Quickstart
```
pip install ydata-synthetic
```

## Examples
Here you can find usage examples of the package and models to synthesize tabular data.

**Credit Fraud dataset**   [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ydataai/ydata-synthetic/blob/master/examples/regular/gan_example.ipynb)

**Stock dataset** [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ydataai/ydata-synthetic/blob/master/examples/timeseries/TimeGAN_Synthetic_stock_data.ipynb)

# Project Resources
- Synthetic GitHub: https://github.com/ydataai/ydata-synthetic
- Synthetic Data Community Slack: [click here to join](http://slack.ydata.ai/)

### In this repo you can find the following GAN architectures:

#### Tabular data
- [GAN](https://arxiv.org/abs/1406.2661)
- [CGAN (Conditional GAN)](https://arxiv.org/abs/1411.1784)
- [WGAN (Wasserstein GAN)](https://arxiv.org/abs/1701.07875)
- [WGAN-GP (Wassertein GAN with Gradient Penalty)](https://arxiv.org/abs/1704.00028)

#### Sequential data
- [TimeGAN](https://papers.nips.cc/paper/2019/file/c9efe5f26cd17ba6216bbe2a7d26d490-Paper.pdf)
