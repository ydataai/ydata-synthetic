# Quickstart

`ydata-synthetic` is equipped to handle both **tabular** (comprising numeric and categorical features) and sequential, **time-series** data. In this section we explain how you can **quickstart the synthesization** of tabular and time-series datasets.

## Synthesizing a Tabular Dataset
The following example showcases how to synthesize the [Adult Census Income](https://www.kaggle.com/datasets/uciml/adult-census-income) dataset with CTGAN:
=== "Tabular Data"
    ```python
        # Import the necessary modules
        from pmlb import fetch_data
        from ydata_synthetic.synthesizers.regular import RegularSynthesizer
        from ydata_synthetic.synthesizers import ModelParameters, TrainParameters

        # Load data
        data = fetch_data('adult')
        num_cols = ['age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week']
        cat_cols = ['workclass','education', 'education-num', 'marital-status',
                    'occupation', 'relationship', 'race', 'sex', 'native-country', 'target']
       
        # Define model and training parameters
        ctgan_args = ModelParameters(batch_size=500, lr=2e-4, betas=(0.5, 0.9))
        train_args = TrainParameters(epochs=501)
       
        # Train the generator model
        synth = RegularSynthesizer(modelname='ctgan', model_parameters=ctgan_args)
        synth.fit(data=data, train_arguments=train_args, num_cols=num_cols, cat_cols=cat_cols)

        # Generate 1000 new synthetic samples
        synth_data = synth.sample(1000) 
    ```

## Synthesizing a Time-Series Dataset
The following example showcases how to synthesize the [Yahoo Stock Price](https://www.kaggle.com/datasets/arashnic/time-series-forecasting-with-yahoo-stock-price) dataset with TimeGAN:
=== "Time-Series Data"
    ```python
        # Import the necessary modules
        import pandas as pd
        from ydata_synthetic.synthesizers.timeseries import TimeSeriesSynthesizer
        from ydata_synthetic.synthesizers import ModelParameters, TrainParameters

        # Define model parameters
        gan_args = ModelParameters(batch_size=128,
                                lr=5e-4,
                                noise_dim=32,
                                layers_dim=128,
                                latent_dim=24,
                                gamma=1)

        train_args = TrainParameters(epochs=50000,
                                    sequence_length=24,
                                    number_sequences=6)

        # Read the data
        stock_data = pd.read_csv("stock_data.csv")

        # Training the TimeGAN synthesizer
        synth = TimeSeriesSynthesizer(modelname='timegan', model_parameters=gan_args)
        synth.fit(stock_data, train_args, num_cols=list(stock_data.columns))

        # Generating new synthetic samples
        synth_data = synth.sample(n_samples=500)
    ```

## Running the Streamlit App
Once the package is [installed](installation.md) with the "streamlit" extra, the app can be launched as:

=== "Streamlit App"
    ```python
        from ydata_synthetic import streamlit_app

        streamlit_app.run()
    ```

The console will then output the URL from which the app can be accessed.

:fontawesome-brands-youtube:{ style="color: #EE0F0F" } Here's a [quick example](https://www.youtube.com/watch?v=6Lzi26szKNo&t=4s) of how to synthesize data with the Streamlit App  – :octicons-clock-24: 5min

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=dd69a9f9-0901-4cb4-9e56-b1e69877dca1" />