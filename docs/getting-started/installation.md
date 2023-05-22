
`ydata-synthetic` is available through PyPi, allowing an easy process of installation and integration with the data science programing environments (Google Colab, Jupyter Notebooks, Visual Studio Code, PyCharm) and stack (`pandas`, `numpy`, `scikit-learn`).

##Installing the package
Currently, the package supports **python versions over 3.9**, and can be installed in Windows, Linux or MacOS operating systems. 

Prior to the package installation, it is recommended the creation of a virtual or `conda` environment:

=== "conda"
    ``` commandline
    conda create -n synth-env python=3.10
    conda activate synth-env
    ```

The above command creates and activates a new environment called "synth-env" with Python version 3.10.X. In the new environment, you can then install `ydata-synthetic`:

=== "pypi"
    ``` commandline
    pip install ydata-synthetic==1.1.0
    ```

:fontawesome-brands-youtube:{ style="color: #EE0F0F" }
[Installing ydata-synthetic](https://www.youtube.com/watch?v=aESmGcxtBdU) – :octicons-clock-24:
5min – Step-by-step installation guide

## Using Google Colab
To install inside a Google Colab notebook, you can use the following:

``` commandline
!pip install ydata-synthetic==1.1.0
```

Make sure your Google Colab is running Python versions `>=3.9, <3.11`. Learn how to configure Python versions on Google Colab [here](https://stackoverflow.com/questions/68657341/how-can-i-update-google-colabs-python-version/68658479#68658479).


## Installing the Streamlit App
Since version 1.0.0, the `ydata-synthetic` includes a GUI experience provided by a Streamlit app. The UI supports the data synthesization process from reading the data to profiling the synthetic data generation, and can be installed as follows:

``` commandline
pip install "ydata-synthetic[streamlit]"
```

Note that Jupyter or Colab Notebooks are not yet supported, so use it in your Python environment.

