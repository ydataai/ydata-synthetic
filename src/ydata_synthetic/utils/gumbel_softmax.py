import tensorflow as tf
from tensorflow.keras import layers
from typing import NamedTuple, Optional

class GumbelSoftmaxLayer(layers.Layer):
    """A Gumbel-Softmax layer implementation that should be stacked on top of a categorical feature logits.

    Arguments:
            tau (float): Temperature parameter of the GS layer
            name (Optional[str]): Name for a single categorical block
    """

    def __init__(self, tau: float, name: Optional[str] = None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.tau = tau

    def call(self, inputs):
        """Computes Gumbel-Softmax for the logits output of a particular categorical feature."""
        noise = gumbel_noise(tf.shape(inputs))
        logits = inputs + noise
        soft_sample = tf.nn.softmax(logits / self.tau, axis=-1)
        hard_sample = tf.stop_gradient(tf.argmax(logits, axis=-1, output_type=tf.int32))
        hard_sample = tf.cast(hard_sample, tf.float32)
        return hard_sample, soft_sample

    def get_config(self):
        config = super().get_config().copy()
        config.update({'tau': self.tau})
        return config

class GumbelSoftmaxActivation(layers.Layer):
    """An interface layer connecting different parts of an incoming tensor to adequate activation functions.
    The tensor parts are qualified according to the passed processor object.
    Processed categorical features are sent to specific Gumbel-Softmax layers.
    Processed features of different kind are sent to a TanH activation.
    Finally all output parts are concatenated and returned in the same order.

    The parts of an incoming tensor are qualified by leveraging a namedtuple pointing to each of the used data \
        processor's pipelines in/out feature maps. For simplicity this object can be taken directly from the data \
        processor col_transform_info."""

    def __init__(self, activation_info: NamedTuple, tau: Optional[float] = None, name: Optional[str] = None, **kwargs):
        """Arguments:
            col_map (NamedTuple): Defines each of the processor pipelines input/output features.
            name (Optional[str]): Name of the GumbelSoftmaxActivation layer
            tau (Optional[float]): Temperature parameter of the GS layer, must be a float bigger than 0"""
        super().__init__(name=name, **kwargs)
        self.tau = 0.2 if not tau else tau  # Defaults to the default value proposed in the original article
        assert isinstance(self.tau, (int, float)) and self.tau > 0, "Optional argument tau must be numerical and \
bigger than 0."

        self._activation_info = activation_info

        self.cat_feats = activation_info.categorical
        self.num_feats = activation_info.numerical

        self._cat_lens = [len([col for col in self.cat_feats.feat_names_out if col.startswith(f'{cat_feat}_')]) \
            for cat_feat in self.cat_feats.feat_names_in]
        self._num_lens = len(self.num_feats.feat_names_out)

    def call(self, inputs):  # pylint: disable=W0221
        num_cols, cat_cols = tf.split(inputs, [self._num_lens, -1], axis=-1)
        cat_cols = tf.split(cat_cols, self._cat_lens if self._cat_lens else [1], axis=-1)

        num_cols = layers.Activation('tanh')(num_cols)
        cat_cols = [GumbelSoftmaxLayer(tau=self.tau)(col)[0] for col in cat_cols]
        return tf.concat([num_cols] + cat_cols, axis=-1)

    def get_config(self):
        config = super().get_config().copy()
        config.update({'activation_info': self._activation_info, 'tau': self.tau})
        return config

def gumbel_noise(shape: tf.TensorShape) -> tf.Tensor:
    """Create a single sample from the standard (loc = 0, scale = 1) Gumbel distribution."""
    uniform_sample = tf.random.uniform(shape, seed=0)
    return -tf.math.log(-tf.math.log(uniform_sample + 1e-20) + 1e-20)
