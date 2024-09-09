![](https://img.shields.io/github/workflow/status/ydataai/ydata-synthetic/prerelease)
![](https://img.shields.io/pypi/status/ydata-synthetic)
[![](https://pepy.tech/badge/ydata-synthetic)](https://pypi.org/project/ydata-synthetic/)
![](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)
[![](https://img.shields.io/pypi/v/ydata-synthetic)](https://pypi.org/project/ydata-synthetic/)
![](https://img.shields.io/github/license/ydataai/ydata-synthetic)
<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=ab07c7a0-c1ee-481e-9368-baf70185cf40" />

<p align="center"><img width="300" src="https://assets.ydata.ai/oss/ydata-synthetic_black.png" alt="YData Synthetic Logo"></p>

Join us on [![Discord](https://img.shields.io/badge/Discord-7289DA?style=for-the-badge&logo=discord&logoColor=white)](https://tiny.ydata.ai/dcai-ydata-synthetic)

# YData Synthetic
`YData-Synthetic` is an open-source package developed in 2020 with the primary goal of educating users about generative models for synthetic data generation. 
Designed as a collection of models, it was intended for exploratory studies and educational purposes. 
However, it was not optimized for the quality, performance, and scalability needs typically required by organizations.

!!! note "Update"
    Even though the journey was fun, and we have learned a lot from the community it is now time to upgrade `ydata-synthetic`.
    Heading towards the future of synthetic data generation we recommend users to transition to `ydata-sdk`, which provides a superior experience with enhanced performance,
    precision, and ease of use, making it the preferred tool for synthetic data generation and a perfect introduction to Generative AI. 

## Synthetic data
### What is synthetic data?
Synthetic data is artificially generated data that is not collected from real world events. It replicates the statistical components of real data without containing any identifiable information, ensuring individuals' privacy.

### Why Synthetic Data?
Synthetic data can be used for many applications:
  - Privacy compliance for data-sharing and Machine Learning development
  - Remove bias
  - Balance datasets
  - Augment datasets

> **Looking for an end-to-end solution to Synthetic Data Generation?**<br>
> [YData Fabric](https://ydata.ai/products/synthetic_data) enables the generation of high-quality datasets within a full UI experience, from data preparation to synthetic data generation and evaluation.<br>
> Check out the [Community Version](https://ydata.ai/register).


## ydata-synthetic to ydata-sdk
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

Are you ready to learn more about synthetic data and the best-practices for synthetic data generation? 
For more materials on [synthetic data generation with Python see the documentation](https://docs.fabric.ydata.ai/latest/sdk/).

## Quickstart
Binary installers for the latest released version are available at the [Python Package Index (PyPI).](https://pypi.org/project/ydata-sdk/)
```commandline
pip install ydata-sdk
```

### The UI guide for synthetic data generation

YData Fabric offers an UI interface to guide you through the steps and inputs to generate structure data.
You can experiment today with [YData Fabric by registering the Community version](https://ydata.ai/register).

### Examples
Here you can find usage examples of the package and models to synthesize tabular data.
  - Tabular [synthetic data generation on Titanic Kaggle dataset](https://github.com/ydataai/ydata-sdk/blob/main/examples/synthesizers/regular_quickstart.py)
  - Time Series [synthetic data generation]('https://github.com/ydataai/ydata-sdk/blob/main/examples/synthesizers/time_series_quickstart.py')
  - More examples are continuously added and can be found in [examples directory](https://github.com/ydataai/ydata-sdk/tree/main/examples).

### Datasets for you to experiment
Here are some example datasets for you to try with the synthesizers:
#### Tabular datasets
- [Adult Census Income](https://www.kaggle.com/datasets/uciml/adult-census-income)
- [Credit card fraud](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- [Cardiovascular Disease dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)

#### Sequential datasets
- [Stock data](https://github.com/ydataai/ydata-synthetic/tree/master/data)
- [FCC MBA data](https://github.com/ydataai/ydata-synthetic/tree/master/data)

## Project Resources

Find below useful literature of how to generate synthetic data and available generative models:

### Tabular data
  - [GAN](https://arxiv.org/abs/1406.2661)
  - [CGAN (Conditional GAN)](https://arxiv.org/abs/1411.1784)
  - [WGAN (Wasserstein GAN)](https://arxiv.org/abs/1701.07875)
  - [WGAN-GP (Wassertein GAN with Gradient Penalty)](https://arxiv.org/abs/1704.00028)
  - [DRAGAN (On Convergence and stability of GANS)](https://arxiv.org/pdf/1705.07215.pdf)
  - [Cramer GAN (The Cramer Distance as a Solution to Biased Wasserstein Gradients)](https://arxiv.org/abs/1705.10743)
  - [CWGAN-GP (Conditional Wassertein GAN with Gradient Penalty)](https://cameronfabbri.github.io/papers/conditionalWGAN.pdf)
  - [CTGAN (Conditional Tabular GAN)](https://arxiv.org/pdf/1907.00503.pdf)
  - [Gaussian Mixture](https://towardsdatascience.com/gaussian-mixture-models-explained-6986aaf5a95)

### Sequential data
  - [TimeGAN](https://papers.nips.cc/paper/2019/file/c9efe5f26cd17ba6216bbe2a7d26d490-Paper.pdf)
  - [DoppelGANger](https://dl.acm.org/doi/pdf/10.1145/3419394.3423643)


## Support
For support in using this library, please join our Discord server. Our Discord community is very friendly and great about quickly answering questions about the use and development of the library. [Click here to join our Discord community!](https://tiny.ydata.ai/dcai-ydata-synthetic)

## FAQs
Have a question? Check out the [Frequently Asked Questions](https://ydata.ai/resources/10-most-asked-questions-on-ydata-synthetic) about `ydata-synthetic`. If you feel something is missing, feel free to [book a beary informal chat with us](https://meetings.hubspot.com/fabiana-clemente).

## License
[MIT License](https://github.com/ydataai/ydata-synthetic/blob/master/LICENSE)
