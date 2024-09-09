# Synthesize tabular data

!!! note "Outdated"
    Note that this example won't work with the latest version of `ydata-synthetic`. 

    Please check `ydata-sdk` to see [how to generate conditional synthetic data](https://docs.fabric.ydata.ai/latest/sdk/examples/synthesize_with_conditional_sampling/).

**Using *CGAN* to generate tabular synthetic data:**

Real-world domains are often described by **tabular data** i.e., data that can be structured and organized in a table-like format, where **features/variables** are represented in **columns**, whereas **observations** correspond to the **rows**.

CGAN is a deep learning model that combines GANs with conditional models to generate data samples based on specific conditions:

- 📑 **Paper:** [Conditonal Generative Adversarial Nets](https://arxiv.org/abs/1411.1784)

Here’s an example of how to synthetize tabular data with CGAN using the [Credit Card](https://www.openml.org/search?type=data&sort=runs&id=1597&status=active) dataset:


```python
--8<-- "examples/regular/models/creditcard_cgan.py"
```
