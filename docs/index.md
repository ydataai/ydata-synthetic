<p></p>
<p align="center"><img width="300" src="https://assets.ydata.ai/oss/ydata-synthetic_black.png" alt="YData Synthetic Logo"></p>
<p></p>

[![pypi](https://img.shields.io/pypi/v/ydata-synthetic)](https://pypi.org/project/ydata-synthetic)
![Pythonversion](https://img.shields.io/badge/python-3.9%20%7C%203.10-blue)
[![downloads](https://static.pepy.tech/badge/ydata-synthetic/month)](https://pepy.tech/project/ydata-synthetic)
![](https://img.shields.io/github/license/ydataai/ydata-synthetic)
![](https://img.shields.io/pypi/status/ydata-synthetic)
[![Build Status](https://github.com/ydataai/ydata-synthetic/actions/workflows/tests.yml/badge.svg?branch=master)](https://github.com/ydataai/ydata-synthetic/actions/workflows/tests.yml)
[![Code Coverage](https://codecov.io/gh/ydataai/ydata-synthetic/branch/master/graph/badge.svg?token=gMptB4YUnF)](https://codecov.io/gh/ydataai/ydata-synthetic)
[![GitHub stars](https://img.shields.io/github/stars/ydataai/ydata-synthetic?style=social)](https://github.com/ydataai/ydata-synthetic)
[![Discord](https://img.shields.io/discord/1037720091376238592?label=Discord&logo=Discord)](https://discord.com/invite/mw7xjJ7b7s)

## Overview
`ydata-synthetic` is the go-to Python package for **synthetic data generation for tabular and time-series data**. It uses the latest Generative AI models to learn the properties of real data and create realistic synthetic data. This project was created to educate the community about synthetic data and its applications in real-world domains, such as data augmentation, bias mitigation, data sharing, and privacy engineering. To learn more about Synthetic Data and its applications, [check this article](https://ydata.ai/resources/10-most-frequently-asked-questions-about-synthetic-data).

## Current Functionality
- 🤖 **Create Realistic Synthetic Data using Generative AI Models:** `ydata-synthetic` supports the state-of-the-art generative adversarial networks for data generation, namely Vanilla GAN, CGAN, WGAN, WGAN-GP, DRAGAN, Cramer GAN, CWGAN-GP, CTGAN, and TimeGAN. Learn more about the use of [GANs for Synthetic Data generation](https://medium.com/ydata-ai/generating-synthetic-tabular-data-with-gans-part-1-866705a77302). 

- 📀 **Synthetic Data Generation for Tabular and Time-Series Data:** The package supports the synthesization of tabular and time-series data, covering a wide range of real-world applications. Learn how to leverage `ydata-synthetic` for [tabular](https://ydata.ai/resources/gans-for-synthetic-data-generation) and [time-series](https://towardsdatascience.com/synthetic-time-series-data-a-gan-approach-869a984f2239) data.

- 💻 **Best Generation Experience in Open Source:** Including a guided UI experience for the generation of synthetic data, from reading the data to visualization of synthetic data. All served by a slick Streamlit app. 
:fontawesome-brands-youtube:{ style="color: #EE0F0F" } Here's a [quick overview](https://www.youtube.com/watch?v=ep0PhwsFx0A) – :octicons-clock-24: 1min

!!! question

    **Looking for an end-to-end solution to Synthetic Data Generation?**

    [YData Fabric](https://ydata.ai/products/synthetic_data) enables the generation of high-quality datasets within a full UI experience, from data preparation to synthetic data generation and evaluation. Check out the [Community Version](https://ydata.ai/ydata-fabric-free-trial).

## Supported Data Types
    
=== "Tabular Data"
    **Tabular data** does not have a temporal dependence, and can be structured and organized in a table-like format, where **features are represented in columns**, whereas **observations correspond to the rows**. 

    Additionally, tabular data usually comprises both *numeric* and *categorical* features. **Numeric** features are those that encode **quantitative** values, whereas **categorical** represent **qualitative** measurements. Categorical features can further divided in *ordinal*, *binary* or *boolean*, and *nominal* features.
    
    Learn more about synthesizing tabular data in this [article](https://ydata.ai/resources/gans-for-synthetic-data-generation), or check the [quickstart guide](getting-started/quickstart.md#synthesizing-a-tabular-dataset) to get started with the synthesization of tabular datasets.

=== "Time-Series Data"
    **Time-series data** exhibit a sequencial, **temporal dependency** between records, and may present a wide range of patterns and trends, including **seasonality** (patterns that repeat at calendar periods -- days, weeks, months -- such as holiday sales, for instance) or **periodicity** (patterns that repeat over time).

    Read more about generating time-series data in this [article](https://ydata.ai/resources/synthetic-time-series-data-a-gan-approach) and check this [quickstart guide](getting-started/quickstart.md#synthesizing-a-time-series-dataset) to get started with time-series data synthesization.
   

## Supported Generative AI Models
The following architectures are currently supported:

- [GAN](https://arxiv.org/abs/1406.2661)
- [CGAN](https://arxiv.org/abs/1411.1784) (Conditional GAN)
- [WGAN](https://arxiv.org/abs/1701.07875) (Wasserstein GAN)
- [WGAN-GP](https://arxiv.org/abs/1704.00028) (Wassertein GAN with Gradient Penalty)
- [DRAGAN](https://arxiv.org/pdf/1705.07215.pdf) (Deep Regret Analytic GAN)
- [Cramer GAN](https://arxiv.org/abs/1705.10743) (Cramer Distance Solution to Biased Wasserstein Gradients)
- [CWGAN-GP](https://cameronfabbri.github.io/papers/conditionalWGAN.pdf) (Conditional Wassertein GAN with Gradient Penalty)
- [CTGAN](https://arxiv.org/pdf/1907.00503.pdf) (Conditional Tabular GAN)
- [TimeGAN](https://papers.nips.cc/paper/2019/file/c9efe5f26cd17ba6216bbe2a7d26d490-Paper.pdf) (specifically for *time-series* data)
