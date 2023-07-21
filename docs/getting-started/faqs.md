# Frequently Asked Questions


## How to get accurate data from my synthetic data generation processes?
Depending on your use case, the downstream application of your synthetic data, and the characteristics of your original data, you will need to adjust your synthetisation process accordingly. That often involves performing a thorough data preparation and fitting your generation models appropriately.

!!! tip

    For a use-case oriented UI experience, try [YData Fabric](https://ydata.ai/products). From an interactive and complete data profiling to an efficient synthetization, your data preparation process will be seamlessly adjusted to your data characteristics.

## What is the best way to evaluate the quality of my synthetic data?
The most appropriate metrics to evaluate the quality of your synthetic data are also dependent on the goal for which synthetic data will be used. Nevertheless, we may define three essential pillars for synthetic data quality: privacy, fidelity, and utility:

- Privacy refers to the ability of synthetic data to withhold any personal, private, or sensitive information, avoiding connections being drawn to the original data and preventing data leakage;

- Fidelity concerns the ability of the new data to preserve the properties of the original data (in other words, it refers to "how faithful, how precise" is the synthetic data in comparison to real data);

- Finally, utility relates to the downstream application where the synthetic data will be used: if the synthetization process is successful, the same insights should be derived from the new data as from the original data. 

For each of these components, several specific statistical measures can be evaluated. 

!!! abstract

    To learn more about how to define specific trade-offs between privacy, fidelity, and utility, check out this white paper on [Synthetic Data Quality Metrics](https://ydata.ai/synthetic-data-quality-metrics).


## Does TimeGAN replicate my full sequence of data?
No. This is an unrealistic expectation because the TimeGAN architecture is not meant to replicate the long-term behavior of your data. 

TimeGAN works with the concept of "windows": it learns to map the data distribution of short-term frames of time, within the time windows you provide. It also considers that those windows are independent of each other, so it cannot return a temporal pattern most people expect. 

That's not supported by this architecture itself, but there are others that allow for both short-term and long-term synthesization, as those available in [YData Fabric](https://ydata.ai/products/synthetic_data). 

!!! abstract

    Learn more about how YData's Time-Series Synthetic Data Generation compare to TimeGAN in [this dedicated post](https://ydata.ai/resources/the-best-generative-ai-model-for-time-series-synthetic-data-generation).

    
# Additional Support
Couldn't find what you need? Reach out to our [dedicated team](https://meetings.hubspot.com/fabiana-clemente) for a quick and *syn-ple* assistance! 