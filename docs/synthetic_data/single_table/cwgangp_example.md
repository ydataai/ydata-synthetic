# Synthesize tabular data

!!! note "Outdated"
    Note that this example won't work with the latest version of `ydata-synthetic`. 

    Please check `ydata-sdk` to see [how to generate conditional synthetic data](https://docs.fabric.ydata.ai/latest/sdk/examples/synthesize_with_conditional_sampling/).

**Using *CWGAN-GP* to generate tabular synthetic data:**

Real-world domains are often described by **tabular data** i.e., data that can be structured and organized in a table-like format, where **features/variables** are represented in **columns**, whereas **observations** correspond to the **rows**.

CWGAN GP is a variant of GAN that incorporates conditional information to generate data samples, while leveraging the Wasserstein distance to improve training stability and sample quality:

- 📑 **Paper:** [Conditional Wasserstein Generative Adversarial Networks](https://cameronfabbri.github.io/papers/conditionalWGAN.pdf)

Here’s an example of how to synthetize tabular data with CWGAN-GP using the [Credit Card](https://www.openml.org/search?type=data&sort=runs&id=1597&status=active) dataset:


```python
--8<-- "examples/regular/models/creditcard_cramergan.py"
```
