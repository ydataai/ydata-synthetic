"Test suite for the Gumbel-Softmax layer implementation."
import tensorflow as tf
from numpy import amax, amin, isclose, ones
from numpy import sum as npsum
from pytest import fixture
from tensorflow.keras import layers

from ydata_synthetic.utils.gumbel_softmax import GumbelSoftmaxLayer


# pylint:disable=W0613
def custom_initializer(shape_list, dtype):
    "A constant weight intializer to ensure test reproducibility."
    return tf.constant(ones((5, 5)), dtype=tf.dtypes.float32)

@fixture(name='rand_input')
def fixture_rand_input():
    "A random, reproducible, input for the mock model."
    return tf.constant(tf.random.normal([4, 5], seed=42))

def test_hard_sample_output_format(rand_input):
    """Tests that the hard output samples are in the expected formats.
    The hard sample should be returned as a one-hot tensor."""
    affined = layers.Dense(5, use_bias = False, kernel_initializer=custom_initializer)(rand_input)
    hard_sample, _ = GumbelSoftmaxLayer()(affined)
    assert npsum(hard_sample) == hard_sample.shape[0], "The sum of the hard samples should equal the number."
    assert all(npsum(hard_sample == 0, 1) == hard_sample.shape[1] - 1), "The hard samples is not a one-hot tensor."

def test_soft_sample_output_format(rand_input):
    """Tests that the soft output samples are in the expected formats.
    The soft sample should be returned as a probabilities tensor."""
    affined = layers.Dense(5, use_bias = False, kernel_initializer=custom_initializer)(rand_input)
    _, soft_sample = GumbelSoftmaxLayer(tau=0.5)(affined)
    assert isclose(npsum(soft_sample), soft_sample.shape[0]), "The sum of the soft samples should be close to \
        the number of records."
    assert amax(soft_sample) <= 1, "Invalid probability values found."
    assert amin(soft_sample) >= 0, "Invalid probability values found."

def test_gradients(rand_input):
    "Performs basic numerical assertions on the gradients of the sof/hard samples."
    def mock(i):
        return GumbelSoftmaxLayer()(layers.Dense(5, use_bias=False, kernel_initializer=custom_initializer)(i))
    with tf.GradientTape() as hard_tape:
        hard_tape.watch(rand_input)
        hard_sample, _ = mock(rand_input)
    with tf.GradientTape() as soft_tape:
        soft_tape.watch(rand_input)
        _, soft_sample = mock(rand_input)
    hard_grads = hard_tape.gradient(hard_sample, rand_input)
    soft_grads = soft_tape.gradient(soft_sample, rand_input)

    assert hard_grads is None, "The hard sample must not compute gradients."
    assert soft_grads is not None, "The soft sample is expected to compute gradients."
    assert npsum(abs(soft_grads)) != 0, "The soft sample is expected to have non-zero gradients."
