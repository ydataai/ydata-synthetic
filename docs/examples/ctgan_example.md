# Synthesize tabular data

**Using *CTGAN* to generate tabular synthetic data:**

Real-world domains are often described by **tabular data** i.e., data that can be structured and organized in a table-like format, where **features/variables** are represented in **columns**, whereas **observations** correspond to the **rows**.

Additionally, real-world data usually comprises both **numeric** and **categorical** features. Numeric features are those that encode quantitative values, whereas categorical represent qualitative measurements.

CTGAN was specifically designed to deal with the challenges posed by tabular datasets, handling mixed (numeric and categorical) data:

- 📑 **Paper:** [Modeling Tabular Data using Conditional GAN](https://arxiv.org/pdf/1907.00503.pdf)

Here’s an example of how to synthetize tabular data with CTGAN using the [Adult Census Income](https://www.kaggle.com/datasets/uciml/adult-census-income?resource=download) dataset:


```python
--8<-- "examples/regular/models/adult_ctgan.py"
```
