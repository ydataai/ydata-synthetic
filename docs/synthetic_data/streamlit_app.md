# The UI guided experience for Synthetic Data generation

´ydata-synthetic´ offers a UI interface to guide you through the steps and inputs to generate structure tabular data. 
The streamlit app is available from *v1.0.0* onwards, and supports the following flows:

- Train a synthesizer model for a single table dataset
- Generate & profile the generated synthetic samples

<p style="text-align:center;">
<iframe width="560" height="315" src="https://www.youtube.com/embed/ep0PhwsFx0A?si=a4UtCbetGdHb7py0" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
</p>

## Installation

pip install ydata-synthetic[streamlit]

## Quickstart

Use the code snippet below in a python file:

!!! warning "Use python scripts"

    I know you probably love Jupyter Notebooks or Google Colab, but make sure that you start your
    synthetic data generation streamlit app from a python script as notebooks are not supported! 

``` py
    from ydata_synthetic import streamlit_app
    streamlit_app.run()
```

Or use the file streamlit_app.py that can be found in the [examples folder]().

``` py
    python -m streamlit_app
```

The below models are supported:

- [ydata-sdk Synthetic Data generator](https://docs.sdk.ydata.ai/0.6/examples/synthesize_tabular_data/)
- CGAN
- WGAN
- WGANGP
- DRAGAN
- CRAMER
- CTGAN

