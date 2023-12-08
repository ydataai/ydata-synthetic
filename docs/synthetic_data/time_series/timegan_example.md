# Synthesize time-series data

**Using *TimeGAN* to generate synthetic time-series data:**

Although tabular data may be the most frequently discussed type of data, a great number of real-world domains — from traffic and daily trajectories to stock prices and energy consumption patterns — produce **time-series data** which introduces several aspects of complexity to synthetic data generation.

Time-series data is structured sequentially, with observations **ordered chronologically** based on their associated timestamps or time intervals. It explicitly incorporates the temporal aspect, allowing for the analysis of trends, seasonality, and other dependencies over time. 

TimeGAN is a model that uses a Generative Adversarial Network (GAN) framework to generate synthetic time series data by learning the underlying temporal dependencies and characteristics of the original data:

- 📑 **Paper:** [Time-series Generative Adversarial Networks](https://papers.nips.cc/paper/2019/file/c9efe5f26cd17ba6216bbe2a7d26d490-Paper.pdf)

Here’s an example of how to synthetize time-series data with TimeGAN using the [Yahoo Stock Price](https://www.kaggle.com/datasets/arashnic/time-series-forecasting-with-yahoo-stock-price) dataset:


```python
--8<-- "examples/timeseries/stock_timegan.py"
```



