# Synthesize tabular data

**Using *DRAGAN* to generate tabular synthetic data:**

Real-world domains are often described by **tabular data** i.e., data that can be structured and organized in a table-like format, where **features/variables** are represented in **columns**, whereas **observations** correspond to the **rows**.

DRAGAN is a GAN variant that uses a gradient penalty to improve training stability and mitigate mode collapse:

- 📑 **Paper:** [On Convergence and Stability of GANs](https://arxiv.org/pdf/1705.07215.pdf)

Here’s an example of how to synthetize tabular data with DRAGAN using the [Adult Census Income](https://www.kaggle.com/datasets/uciml/adult-census-income?resource=download) dataset:


```python
--8<-- "examples/regular/models/adult_dragan.py"
```
