<p></p>
<p align="center"><img width="300" src="https://assets.ydata.ai/oss/ydata-synthetic_black.png" alt="YData Synthetic Logo"></p>
<p></p>

[![pypi](https://img.shields.io/pypi/v/ydata-synthetic)](https://pypi.org/project/ydata-synthetic)
![Pythonversion](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)
[![downloads](https://static.pepy.tech/badge/ydata-synthetic/month)](https://pepy.tech/project/ydata-synthetic)
![](https://img.shields.io/github/license/ydataai/ydata-synthetic)
![](https://img.shields.io/pypi/status/ydata-synthetic)
[![Build Status](https://github.com/ydataai/ydata-synthetic/actions/workflows/tests.yml/badge.svg?branch=master)](https://github.com/ydataai/ydata-synthetic/actions/workflows/tests.yml)
[![Code Coverage](https://codecov.io/gh/ydataai/ydata-synthetic/branch/master/graph/badge.svg?token=gMptB4YUnF)](https://codecov.io/gh/ydataai/ydata-synthetic)
[![GitHub stars](https://img.shields.io/github/stars/ydataai/ydata-synthetic?style=social)](https://github.com/ydataai/ydata-synthetic)
[![Discord](https://img.shields.io/discord/1037720091376238592?label=Discord&logo=Discord)](https://discord.com/invite/mw7xjJ7b7s)

## Overview
`YData-Synthetic` is an open-source package developed in 2020 with the primary goal of educating users about generative models for synthetic data generation. 
Designed as a collection of models, it was intended for exploratory studies and educational purposes. 
However, it was not optimized for the quality, performance, and scalability needs typically required by organizations.

!!! tip "We are now ydata-sdk!"
    Even though the journey was fun, and we have learned a lot from the community it is now time to upgrade `ydata-synthetic`.

    Heading towards the future of synthetic data generation we recommend users to transition to `ydata-sdk`, which provides a superior experience with enhanced performance,
    precision, and ease of use, making it the preferred tool for synthetic data generation and a perfect introduction to Generative AI. 

## Supported Data Types
    
=== "Tabular Data"
    **Tabular data** does not have a temporal dependence, and can be structured and organized in a table-like format, where **features are represented in columns**, whereas **observations correspond to the rows**. 

    Additionally, tabular data usually comprises both *numeric* and *categorical* features. **Numeric** features are those that encode **quantitative** values, whereas **categorical** represent **qualitative** measurements. Categorical features can further divided in *ordinal*, *binary* or *boolean*, and *nominal* features.
    
    Learn more about synthesizing tabular data in this [article](https://ydata.ai/resources/gans-for-synthetic-data-generation), or check the [quickstart guide](getting-started/quickstart.md#synthesizing-a-tabular-dataset) to get started with the synthesization of tabular datasets.

=== "Time-Series Data"
    **Time-series data** exhibit a sequencial, **temporal dependency** between records, and may present a wide range of patterns and trends, including **seasonality** (patterns that repeat at calendar periods -- days, weeks, months -- such as holiday sales, for instance) or **periodicity** (patterns that repeat over time).

    Read more about generating [time-series data in this article](https://ydata.ai/resources/synthetic-time-series-data-a-gan-approach) and check this [quickstart guide](getting-started/quickstart.md#synthesizing-a-time-series-dataset) to get started with time-series data synthesization.

=== "Multi-Table Data"
    **Multi-Table data** or databases exhibit a referential behaviour between and database schema that is expected to be replicated and respected by the synthetic data generated. 
    Read more about database [synthetic data generation in this article]() and check this [quickstart guide for Multi-Table synthetic data generation]()
    **Time-series data** exhibit a sequential, **temporal dependency** between records, and may present a wide range of patterns and trends, including **seasonality** (patterns that repeat at calendar periods -- days, weeks, months -- such as holiday sales, for instance) or **periodicity** (patterns that repeat over time).

## Validate the quality of your synthetic data generated

Validating the quality of synthetic data is essential to ensure its usefulness and privacy. YData Fabric provides tools for comprehensive synthetic data evaluation through:

1. **Profile Comparison Visualization:**
Fabric delivers side-by-side visual comparisons of key data properties (e.g., distributions, correlations, and outliers) between synthetic and original datasets, allowing users to assess fidelity at a glance.

2. **PDF Report with Metrics:**
Fabric generates a PDF report that includes key metrics to evaluate:

- Fidelity: How closely synthetic data matches the original.
- Utility: How well it performs in real-world tasks.
- Privacy: Risk assessment of data leakage and re-identification.

These tools ensure a thorough validation of synthetic data quality, making it reliable for real-world use.

## Supported Generative AI Models
With the upcoming update of `ydata-synthetic`to `ydata-sdk`, users will now have access to a single API that automatically selects and optimizes
the best generative model for their data. This streamlined approach eliminates the need to choose between
various models manually, as the API intelligently identifies the optimal model based on the specific dataset and use case.

Instead of having to manually select from models such as:

- [GAN](https://arxiv.org/abs/1406.2661)
- [CGAN](https://arxiv.org/abs/1411.1784) (Conditional GAN)
- [WGAN](https://arxiv.org/abs/1701.07875) (Wasserstein GAN)
- [WGAN-GP](https://arxiv.org/abs/1704.00028) (Wassertein GAN with Gradient Penalty)
- [DRAGAN](https://arxiv.org/pdf/1705.07215.pdf) (Deep Regret Analytic GAN)
- [Cramer GAN](https://arxiv.org/abs/1705.10743) (Cramer Distance Solution to Biased Wasserstein Gradients)
- [CWGAN-GP](https://cameronfabbri.github.io/papers/conditionalWGAN.pdf) (Conditional Wassertein GAN with Gradient Penalty)
- [CTGAN](https://arxiv.org/pdf/1907.00503.pdf) (Conditional Tabular GAN)
- [TimeGAN](https://papers.nips.cc/paper/2019/file/c9efe5f26cd17ba6216bbe2a7d26d490-Paper.pdf) (specifically for *time-series* data)
- [DoppelGANger](https://dl.acm.org/doi/pdf/10.1145/3419394.3423643) (specifically for *time-series* data)

The new API handles model selection automatically, optimizing for the best performance in fidelity, utility, and privacy.
This significantly simplifies the synthetic data generation process, ensuring that users get the highest quality output without
the need for manual intervention and tiring hyperparameter tuning. 

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=dd69a9f9-0901-4cb4-9e56-b1e69877dca1" />