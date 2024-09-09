# Synthesize time-series data

!!! note "Outdated"
    Note that this example won't work with the latest version of `ydata-synthetic`. 

    Please check `ydata-sdk` to see [how to generate synthetic time-series data](https://docs.fabric.ydata.ai/latest/sdk/examples/synthesize_timeseries_data/).


**Using *DoppelGANger* to generate synthetic time-series data:**

Although tabular data may be the most frequently discussed type of data, a great number of real-world domains — from traffic and daily trajectories to stock prices and energy consumption patterns — produce **time-series data** which introduces several aspects of complexity to synthetic data generation.

Time-series data is structured sequentially, with observations **ordered chronologically** based on their associated timestamps or time intervals. It explicitly incorporates the temporal aspect, allowing for the analysis of trends, seasonality, and other dependencies over time. 

DoppelGANger is a model that uses a Generative Adversarial Network (GAN) framework to generate synthetic time series data by learning the underlying temporal dependencies and characteristics of the original data:

- 📑 **Paper:** [Using GANs for Sharing Networked Time Series Data: Challenges, Initial Promise, and Open Questions](https://dl.acm.org/doi/pdf/10.1145/3419394.3423643)

Here’s an example of how to synthetize time-series data with DoppelGANger using the [Measuring Broadband America](https://www.fcc.gov/reports-research/reports/measuring-broadband-america/raw-data-measuring-broadband-america-seventh) dataset:


```python
--8<-- "examples/timeseries/mba_doppelganger.py"
```



