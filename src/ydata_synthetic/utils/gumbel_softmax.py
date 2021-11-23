"""Gumbel-Softmax layer implementation.
Reference: https://arxiv.org/pdf/1611.04051.pdf"""
from typing import Optional

# pylint: disable=E0401
from tensorflow import (Tensor, TensorShape, concat, one_hot, split, squeeze,
                        stop_gradient)
from tensorflow.keras.layers import Activation, Layer
from tensorflow.math import log
from tensorflow.nn import softmax
from tensorflow.random import categorical, uniform

from ydata_synthetic.preprocessing.base_processor import ProcessorInfo

TOL = 1e-20


def gumbel_noise(shape: TensorShape) -> Tensor:
    """Create a single sample from the standard (loc = 0, scale = 1) Gumbel distribution."""
    uniform_sample = uniform(shape)
    return -log(-log(uniform_sample + TOL) + TOL)


class GumbelSoftmaxLayer(Layer):
    "A Gumbel-Softmax layer implementation that should be stacked on top of a categorical feature logits."

    def __init__(self, tau: float = 0.2, name: Optional[str] = None):
        super().__init__(name = name)
        self.tau = tau

    # pylint: disable=W0221, E1120
    def call(self, _input):
        """Computes Gumbel-Softmax for the logits output of a particular categorical feature."""
        noised_input = _input + gumbel_noise(_input.shape)
        soft_sample = softmax(noised_input/self.tau, -1)
        hard_sample = stop_gradient(squeeze(one_hot(categorical(log(soft_sample), 1), _input.shape[-1]), 1))
        return hard_sample, soft_sample


class ActivationInterface(Layer):
    """An interface layer connecting different parts of an incoming tensor to adequate activation functions.
    The tensor parts are qualified according to the passed processor object.
    Processed categorical features are sent to specific Gumbel-Softmax layers.
    Processed features of different kind are sent to a TanH activation.
    Finally all output parts are concatenated and returned in the same order.

    The parts of an incoming tensor are qualified by leveraging a data processor object (from the synthesizer)."""

    def __init__(self, metadata: ProcessorInfo, name: Optional[str] = None):
        """Arguments:
            metadata (ProcessorInfo): A metadata object defining the processor pipelines input/output features.
            name (Optional[str]): Name of the layer"""
        super().__init__(name)

        self._cat_names = metadata.categorical.feat_names_in_
        self._num_names = metadata.numerical.feat_names_in_

        self._cat_lens = [len([col for col in metadata.categorical.feat_names_out \
            if ''.join(col.split('_')[:-1]) == cat_feat]) for cat_feat in self._cat_names]
        self._num_lens = len(metadata.numerical.feat_names_out)

    def call(self, _input):  # pylint: disable=W0221
        num_cols, cat_cols = split(_input, [self._num_lens, -1], 1, name='split_num_cats')
        cat_cols = split(cat_cols, self._cat_lens, 1, name='split_cats')
        num_cols = [Activation('tanh', name='num_cols_activation')(num_cols)]
        cat_cols = [GumbelSoftmaxLayer(name=name).call(cat_col)[0] for name, cat_col in zip(self._cat_names, cat_cols)]
        return concat(num_cols+cat_cols, 1)
