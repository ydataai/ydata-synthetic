
`ydata-sdk` is available through PyPi, allowing an easy process of installation and integration with the data science programing environments (Google Colab, Jupyter Notebooks, Visual Studio Code, PyCharm) and stack (`pandas`, `numpy`, `scikit-learn`).

##Installing the package
Currently, the package supports **python versions over 3.9 and up-to python 3.12**, and can be installed in Windows, Linux or MacOS operating systems. 

Prior to the package installation, it is recommended the creation of a virtual or `conda` environment:

=== "conda"
    ``` commandline
    conda create -n synth-env python=3.12
    conda activate synth-env
    ```

The above command creates and activates a new environment called "synth-env" with Python version 3.12.X. In the new environment, you can then install `ydata-sdk`:

=== "pypi"
    ``` commandline
    pip install ydata-sdk
    ```

:fontawesome-brands-youtube:{ style="color: #EE0F0F" }
[Installing ydata-synthetic](https://www.youtube.com/watch?v=aESmGcxtBdU) – :octicons-clock-24:
5min – Step-by-step installation guide

## Using Google Colab
To install inside a Google Colab notebook, you can use the following:

``` commandline
!pip install ydata-sdk
```

Make sure your Google Colab is running Python versions `>=3.9, <=3.12`. Learn how to configure Python versions on Google Colab [here](https://stackoverflow.com/questions/68657341/how-can-i-update-google-colabs-python-version/68658479#68658479).
