# Frequently Asked Questions

## How to get accurate data from my synthetic data generation processes?
Depending on your use case, the downstream application of your synthetic data, and the characteristics of your original data, you will need to adjust your synthetisation process accordingly. That often involves performing a thorough data preparation and fitting your generation models appropriately.

!!! tip

    For a use-case oriented UI experience, try [YData Fabric](https://ydata.ai/ydata-fabric-free-trial). From an interactive and complete data profiling to an efficient synthetization, your data preparation process will be seamlessly adjusted to your data characteristics.

## What is the best way to evaluate the quality of my synthetic data?
The most appropriate metrics to evaluate the quality of your synthetic data are also dependent on the goal for which synthetic data will be used. Nevertheless, we may define three essential pillars for synthetic data quality: privacy, fidelity, and utility:

- Privacy refers to the ability of synthetic data to withhold any personal, private, or sensitive information, avoiding connections being drawn to the original data and preventing data leakage;

- Fidelity concerns the ability of the new data to preserve the properties of the original data (in other words, it refers to "how faithful, how precise" is the synthetic data in comparison to real data);

- Finally, utility relates to the downstream application where the synthetic data will be used: if the synthetization process is successful, the same insights should be derived from the new data as from the original data. 

For each of these components, several specific statistical measures can be evaluated. 

!!! abstract

    To learn more about how to define specific trade-offs between privacy, fidelity, and utility, check out this white paper on [Synthetic Data Quality Metrics](https://ydata.ai/synthetic-data-quality-metrics).


## How to generate synthetic data in Google Colab and Python Environments?
Most issues with installations are usually associated with unsupported Python versions or misalignment between python environments and package requirements. 

Let’s see how you can get both right:

### Python Versions
Note that `ydata-sdk` currently requires Python >=3.9, < 3.13 so if you're trying to run our code in Google Colab, then you need to [update your Google Colab’s Python version](https://stackoverflow.com/questions/68657341/how-can-i-update-google-colabs-python-version/68658479#68658479) accordingly. The same goes for your development environment.

### Virtual Environments
A lot of troubleshooting arises due to misalignments between environments and package requirements.
Virtual Environments isolate your installations from the "global" environment so that you don't have to worry about conflicts. 

Using conda, creating a new environment is as easy as running this on your shell:

```
conda create --name synth-env python==3.12 pip
conda activate synth-env
pip install ydata-sdk
```

Now you can open up your Python editor or Jupyter Lab and use the synth-env as your development environment, without having to worry about conflicting versions or packages between projects!


## Does TimeGAN replicate my full sequence of data?
No. This is an unrealistic expectation because the TimeGAN architecture is not meant to replicate the long-term behavior of your data. 

TimeGAN works with the concept of "windows": it learns to map the data distribution of short-term frames of time, within the time windows you provide. It also considers that those windows are independent of each other, so it cannot return a temporal pattern most people expect. 

That's not supported by this architecture itself, but there are others that allow for both short-term and long-term synthesization, as those available in [YData Fabric](https://ydata.ai/products/synthetic_data). 

!!! abstract

    Learn more about how YData's Time-Series Synthetic Data Generation compare to TimeGAN in [this dedicated post](https://ydata.ai/resources/the-best-generative-ai-model-for-time-series-synthetic-data-generation).

    
# Additional Support
Couldn't find what you need? Reach out to our [dedicated team](https://meetings.hubspot.com/fabiana-clemente) for a quick and *syn-ple* assistance! 