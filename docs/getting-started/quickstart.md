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
        from ydata_synthetic.synthesizers import ModelParameters
        from ydata_synthetic.synthesizers.timeseries import TimeGAN
        from ydata_synthetic.preprocessing.timeseries.utils import real_data_loading

        # Load and preprocess data
        stock_data_df = pd.read_csv("stock_data.csv")
        processed_data = real_data_loading(stock_data_df.values, seq_len=24)
       
        # Define model and training parameters
        gan_args = ModelParameters(batch_size=128, lr=5e-4, noise_dim=128, layers_dim=128)
        synth = TimeGAN(model_parameters=gan_args, hidden_dim=24, seq_len=24, n_seq=6, gamma=1)

        # Train the generator model
        synth.train(data=processed_data, train_steps=50000)

        # Generate new synthetic data
        synth_data = synth.sample(len(stock_data_df))
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