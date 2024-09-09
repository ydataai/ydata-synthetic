# Synthesize tabular data

!!! note "Outdated"
    Note that this example won't work with the latest version of `ydata-synthetic`. 

    Please check `ydata-sdk` to see [how to generate synthetic data](https://docs.fabric.ydata.ai/latest/sdk/examples/synthesize_tabular_data/).

**Using *WGAN-GP* to generate tabular synthetic data:**

Real-world domains are often described by **tabular data** i.e., data that can be structured and organized in a table-like format, where **features/variables** are represented in **columns**, whereas **observations** correspond to the **rows**.

WGANGP is a variant of GAN that incorporates a gradient penalty term to enhance training stability and improve the diversity of generated samples:

- 📑 **Paper:** [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028)

Here’s an example of how to synthetize tabular data with WGAN-GP using the [Adult Census Income](https://www.kaggle.com/datasets/uciml/adult-census-income?resource=download) dataset:


```python
--8<-- "examples/regular/models/adult_wgangp.py"
```
