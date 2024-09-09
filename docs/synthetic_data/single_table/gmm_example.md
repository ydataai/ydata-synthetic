# Synthesize tabular data

!!! note "Outdated"
    Note that this example won't work with the latest version of `ydata-synthetic`. 

    Please check `ydata-sdk` to see [how to generate synthetic data](https://docs.fabric.ydata.ai/latest/sdk/examples/synthesize_tabular_data/).

**Using *GMMs* to generate tabular synthetic data:**

Real-world domains are often described by **tabular data** i.e., data that can be structured and organized in a table-like
format, where **features/variables** are represented in **columns**, whereas **observations** correspond to the **rows**.

Gaussian Mixture models (GMMs) are a type of probabilistic models. Probabilistic models can also be leveraged to generate
synthetic data. Particularly, the way GMMs are able to generate synthetic data, is by learning the original data distribution
while fitting it to a mixture of Gaussian distributions.

- 📑 **Blogpost:** [Generate synthetic data with Gaussian Mixture models](https://ydata.ai/resources/synthetic-data-generation-with-gaussian-mixture-models)
- **Google Colab:** [Generate Adult census data with GMM](https://colab.research.google.com/github/ydataai/ydata-synthetic/blob/master/examples/regular/models/Fast_Adult_Census_Income_Data.ipynb)