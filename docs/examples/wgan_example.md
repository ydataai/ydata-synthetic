# Synthesize tabular data

**Using *WGAN* to generate tabular synthetic data:**

Real-world domains are often described by **tabular data** i.e., data that can be structured and organized in a table-like format, where **features/variables** are represented in **columns**, whereas **observations** correspond to the **rows**.

WGAN is a variant of GAN that utilizes the Wasserstein distance to improve training stability and generate higher quality samples:

- 📑 **Paper:** [Wasserstein GAN](https://arxiv.org/abs/1701.07875)

Here’s an example of how to synthetize tabular data with WGAN using the [Credit Card](https://www.openml.org/search?type=data&sort=runs&id=1597&status=active) dataset:


```python
--8<-- "examples/regular/models/creditcard_wgan.py"
```
