# Synthesize tabular data

!!! note "Outdated"
    Note that this example won't work with the latest version of `ydata-synthetic`. 

    Please check `ydata-sdk` to see [how to generate synthetic data](https://docs.fabric.ydata.ai/latest/sdk/examples/synthesize_tabular_data/).

**Using *CTGAN* to generate tabular synthetic data:**

Real-world domains are often described by **tabular data** i.e., data that can be structured and organized in a table-like format, where **features/variables** are represented in **columns**, whereas **observations** correspond to the **rows**.

Additionally, real-world data usually comprises both **numeric** and **categorical** features. Numeric features are those that encode quantitative values, whereas categorical represent qualitative measurements.

CTGAN was specifically designed to deal with the challenges posed by tabular datasets, handling mixed (numeric and categorical) data:

- 📑 **Paper:** [Modeling Tabular Data using Conditional GAN](https://arxiv.org/pdf/1907.00503.pdf)

Here’s an example of how to synthetize tabular data with CTGAN using the [Adult Census Income](https://www.kaggle.com/datasets/uciml/adult-census-income?resource=download) dataset:

```python
--8<-- "examples/regular/models/adult_ctgan.py"
```

## Best practices & results optimization

!!! tip "Generate the best synthetic data quality"
    
    If you are having a hard time in ensuring that CTGAN returns the synthetic data quality that you need for your use-case
    give it a try to [YData Fabric Synthetic Data](https://ydata.ai/register). 
    **Fabric Synthetic Data generation** is considered the best in terms of quality. 
    [Read more about it in this benchmark](https://www.linkedin.com/pulse/generative-ai-synthetic-data-vendor-comparison-best-vincent-granville). 

**CTGAN**, as any other Machine Learning model, requires optimization at the level of the data preparation as well as 
hyperparameter tuning. Here follows a list of best-practices and tips to improve your synthetic data quality:

- **Understand Your Data:**
Thoroughly understand the characteristics and distribution of your original dataset before using CTGAN.
Identify important features, correlations, and patterns in the data.
Leverage [ydata-profiling](https://pypi.org/project/ydata-profiling/) feature to automate the process of understanding your data.

- **Data Preprocess:**
Clean and preprocess your data to handle missing values, outliers, and other anomalies before training CTGAN.
Standardize or normalize numerical features to ensure consistent scales.

- **Feature Engineering:**
Create additional meaningful features that could improve the quality of the synthetic data.

- **Optimize Model Parameters:**
Experiment with CTGAN hyperparameters such as *epochs*, *batch_size*, and *gen_dim* to find the values that work best
for your specific dataset.
Fine-tune the *learning rate* for better convergence.

- **Conditional Generation:**
Leverage the conditional generation capabilities of CTGAN by specifying conditions for certain features if applicable.
Adjust the conditioning mechanism to enhance the relevance of generated samples.

- **Handle Imbalanced Data:**
If your original dataset is imbalanced, ensure that CTGAN captures the distribution of minority classes effectively.
Adjust sampling strategies if needed.

- **Use Larger Datasets:**
Train CTGAN on larger datasets when possible to capture a more comprehensive representation of the underlying data distribution.