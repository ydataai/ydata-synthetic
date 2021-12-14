"GumbelSoftmaxActivation layer test suite."
from itertools import cycle, islice
from re import search

from numpy import array, cumsum, isin, split
from numpy import sum as npsum
from numpy.random import normal
from pandas import DataFrame, concat
from pytest import fixture
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input

from ydata_synthetic.preprocessing.regular.processor import \
    RegularDataProcessor
from ydata_synthetic.utils.gumbel_softmax import GumbelSoftmaxActivation

BATCH_SIZE = 10

@fixture(name='noise_batch')
def fixture_noise_batch():
    "Sample noise for mock output generation."
    return normal(size=(BATCH_SIZE, 16))

@fixture(name='mock_data')
def fixture_mock_data():
    "Creates mock data for the tests."
    num_block = DataFrame(normal(size=(BATCH_SIZE, 6)), columns = [f'num_{i}' for i in range(6)])
    cat_block_1 = DataFrame(array(list(islice(cycle(range(2)), BATCH_SIZE))), columns = ['cat_0'])
    cat_block_2 = DataFrame(array(list(islice(cycle(range(4)), BATCH_SIZE))), columns = ['cat_1'])
    return concat([num_block, cat_block_1, cat_block_2], axis = 1)

@fixture(name='mock_processor')
def fixture_mock_processor(mock_data):
    "Creates a mock data processor for the mock data."
    num_cols = [col for col in mock_data.columns if col.startswith('num')]
    cat_cols = [col for col in mock_data.columns if col.startswith('cat')]
    return RegularDataProcessor(num_cols, cat_cols).fit(mock_data)

# pylint: disable=C0103
@fixture(name='mock_generator')
def fixture_mock_generator(noise_batch, mock_processor):
    "A mock generator with the Activation Interface as final layer."
    input_ = Input(shape=noise_batch.shape[1], batch_size = BATCH_SIZE)
    dim = 15
    data_dim = 12
    x = Dense(dim, activation='relu')(input_)
    x = Dense(dim * 2, activation='relu')(x)
    x = Dense(dim * 4, activation='relu')(x)
    x = Dense(data_dim)(x)
    x = GumbelSoftmaxActivation(activation_info=mock_processor.col_transform_info, name='act_itf')(x)
    return Model(inputs=input_, outputs=x)

@fixture(name='mock_output')
def fixture_mock_output(noise_batch, mock_generator):
    "Returns mock output of the model as a numpy object."
    return mock_generator(noise_batch).numpy()

# pylint: disable=W0632
def test_io(mock_processor, mock_output):
    "Tests the output format of the activation interface for a known input."
    num_lens = len(mock_processor.col_transform_info.numerical.feat_names_out)
    cat_lens = len(mock_processor.col_transform_info.categorical.feat_names_out)
    assert mock_output.shape == (BATCH_SIZE, num_lens + cat_lens), "The output has wrong shape."
    num_part, cat_part = split(mock_output, [num_lens], 1)
    assert not isin(num_part, [0, 1]).all(), "The numerical block is not expected to contain 0 or 1."
    assert isin(cat_part, [0, 1]).all(), "The categorical block is expected to contain only 0 or 1."
    cat_i, cat_o = mock_processor.col_transform_info.categorical
    cat_blocks = cumsum([len([col for col in cat_o if col.startswith(feat) and search('_[0-9]*$', col)]) \
        for feat in cat_i])
    cat_blocks = split(cat_part, cat_blocks[:-1], 1)
    assert all(npsum(abs(block)) == BATCH_SIZE for block in cat_blocks), "There are non one-hot encoded \
        categorical blocks."
