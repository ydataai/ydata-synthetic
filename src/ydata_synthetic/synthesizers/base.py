"""
    Implements a GAN BaseModel synthesizer, not meant to be directly instantiated
"""
from abc import ABC

from typing import List, Optional
from warnings import warn

from collections import namedtuple


_model_parameters = ['batch_size', 'lr', 'betas', 'layers_dim', 'noise_dim',
                     'n_cols', 'seq_len', 'condition', 'n_critic', 'n_features',
                     'tau_gs', 'generator_dims', 'critic_dims', 'l2_scale',
                     'latent_dim', 'gp_lambda', 'pac', 'gamma', 'tanh']
_model_parameters_df = [128, 1e-4, (None, None), 128, 264,
                        None, None, None, 1, None, 0.2, [256, 256],
                        [256, 256], 1e-6, 128, 10.0, 10, 1, False]

_train_parameters = ['cache_prefix', 'label_dim', 'epochs', 'sample_interval',
                     'labels', 'n_clusters', 'epsilon', 'log_frequency',
                     'measurement_cols', 'sequence_length', 'number_sequences',
                     'sample_length', 'rounds']

ModelParameters = namedtuple('ModelParameters', _model_parameters, defaults=_model_parameters_df)
TrainParameters = namedtuple('TrainParameters', _train_parameters, defaults=('', None, 300, 50, None, 10, 0.005, True, None, 1, 1, 1, 1))

class BaseModel(ABC):
    """
    This class is deprecated and should no longer be used.
    Please refer to the new implementation.
    """
    def __init__(self,
                  model_parameters: ModelParameters,
                  num_cols: Optional[List[str]] = None,
                  cat_cols: Optional[List[str]] = None,
                  **kwargs):
        warn(
            f"{self.__class__.__name__} is deprecated. Please leverage ydata-sdk **RegularSynthesizer** or **TimeSeriesSynthesizer** instead. For more information, "
            f"check ydata-sdk documentation:  https://docs.fabric.ydata.ai/latest/sdk/examples/.",
            DeprecationWarning,
            stacklevel=2
        )