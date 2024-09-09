# Synthesize time-series data

!!! note "Outdated"
    Note that this example won't work with the latest version of `ydata-synthetic`. 

    Please check `ydata-sdk` to see [how to generate synthetic time-series data](https://docs.fabric.ydata.ai/latest/sdk/examples/synthesize_timeseries_data/).

## Why YData Fabric vs TimeGAN for time-series data
YData Fabric offers advanced capabilities for time-series synthetic data generation, surpassing TimeGAN in terms of flexibility,
scalability, and ease of use. With YData Fabric, users can generate high-quality synthetic time-series data while benefiting from built-in data profiling tools
that ensure the integrity and consistency of the data. Unlike TimeGAN, which is a single model for time-series, YData Fabric offers a solution that is suitable for different types of datasets and behaviours.
Additionally, YData Fabric is designed for scalability, enabling seamless handling of large, complex time-series datasets. Its guided UI makes it easy to adapt to different time-series scenarios,
from healthcare to financial data, making it a more comprehensive and flexible solution for time-series data generation.

For more on [YData Fabric vs Synthetic data generation with TimeGAN read this blogpost](https://ydata.ai/resources/the-best-generative-ai-model-for-time-series-synthetic-data-generation).

## Using *TimeGAN* to generate synthetic time-series data

Although tabular data may be the most frequently discussed type of data, a great number of real-world domains — from traffic and daily trajectories to stock prices and energy consumption patterns — produce **time-series data** which introduces several aspects of complexity to synthetic data generation.

Time-series data is structured sequentially, with observations **ordered chronologically** based on their associated timestamps or time intervals. It explicitly incorporates the temporal aspect, allowing for the analysis of trends, seasonality, and other dependencies over time. 

TimeGAN is a model that uses a Generative Adversarial Network (GAN) framework to generate synthetic time series data by learning the underlying temporal dependencies and characteristics of the original data:

- 📑 **Paper:** [Time-series Generative Adversarial Networks](https://papers.nips.cc/paper/2019/file/c9efe5f26cd17ba6216bbe2a7d26d490-Paper.pdf)

Here’s an example of how to synthetize time-series data with TimeGAN using the [Yahoo Stock Price](https://www.kaggle.com/datasets/arashnic/time-series-forecasting-with-yahoo-stock-price) dataset:


```python
--8<-- "examples/timeseries/stock_timegan.py"
```



