"""
DoppelGANger architecture example file
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
from ydata_synthetic.synthesizers.timeseries import TimeSeriesSynthesizer
from ydata_synthetic.synthesizers import ModelParameters, TrainParameters

def load_data(file_path="../../data/fcc_mba.csv"):
    """Load data and preprocess it."""
    mba_data = pd.read_csv(file_path)
    numerical_cols = ["traffic_byte_counter", "ping_loss_rate"]
    categorical_cols = [col for col in mba_data.columns if col not in numerical_cols]
    return mba_data, numerical_cols, categorical_cols

def train_model(model_dop_gan, mba_data, train_args, numerical_cols, categorical_cols):
    """Train the DoppelGANger synthesizer."""
    model_dop_gan.fit(mba_data, train_args, num_cols=numerical_cols, cat_cols=categorical_cols)
    return model_dop_gan

def visualize_results(mba_data, synth_df):
    """Visualize the results."""
    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    plt.plot(mba_data['traffic_byte_counter'].reset_index(drop=True), label='Real Traffic')
    plt.plot(synth_df['traffic_byte_counter'].reset_index(drop=True), label='Synthetic Traffic', alpha=0.7)
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Traffic Comparison')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(mba_data['ping_loss_rate'].reset_index(drop=True), label='Real Ping')
    plt.plot(synth_df['ping_loss_rate'].reset_index(drop=True), label='Synthetic Ping', alpha=0.7)
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Ping Comparison')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Importing necessary libraries
    mba_data, numerical_cols, categorical_cols = load_data()

    # Define model parameters
    model_args = ModelParameters(batch_size=100,
                                 lr=0.001,
                                 betas=(0.2, 0.9),
                                 latent_dim=20,
                                 gp_lambda=2,
                                 pac=1)

    train_args = TrainParameters(epochs=400, sequence_length=56,
                                 sample_length=8, rounds=1,
                                 measurement_cols=["traffic_byte_counter", "ping_loss_rate"])

    model_file = 'doppelganger_mba'
    if os.path.exists(model_file):
        model_dop_gan = TimeSeriesSynthesizer.load(model_file)
    else:
        model_dop_gan = TimeSeriesSynthesizer(modelname='doppelganger', model_parameters=model_args)
        model_dop_gan = train_model(model_dop_gan, mba_data, train_args, numerical_cols, categorical_cols)
        model_dop_gan.save(model_file)

    # Generate synthetic data
    synth_data = model_dop_gan.sample(n_samples=600)
    synth_df = pd.concat(synth_data, axis=0)

    # Visualize the results
    visualize_results(mba_data, synth_df)
