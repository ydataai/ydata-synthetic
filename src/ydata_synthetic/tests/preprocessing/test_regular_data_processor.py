"""
Test suite for the RegularProcessor.
"""
from numpy import isclose, ndarray
from pmlb import fetch_data
from pytest import fixture, raises
from sklearn.exceptions import NotFittedError

from ydata_synthetic.preprocessing.regular.processor import \
    RegularDataProcessor


@fixture
def regular_data_example():
    return fetch_data('adult')

@fixture
def regular_data_processor_args(regular_data_example):
    num_cols = ['fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week']
    cat_cols = list(set(regular_data_example.columns).difference(set(num_cols)))
    return num_cols, cat_cols

@fixture
def overlapped_column_lists(regular_data_processor_args):
    num_cols, cat_cols = regular_data_processor_args
    cat_cols.append(num_cols[0])
    return num_cols, cat_cols

@fixture
def incomplete_column_lists(regular_data_processor_args):
    num_cols, cat_cols = regular_data_processor_args
    num_cols.pop()
    return num_cols, cat_cols

@fixture
def regular_data_processor(regular_data_processor_args):
    num_cols, cat_cols = regular_data_processor_args
    return RegularDataProcessor(num_cols=num_cols, cat_cols=cat_cols)

def test_is_fitted(regular_data_processor, regular_data_example):
    "Tests raising NotFittedError in attempting to transform with a non fitted processor."
    with raises(NotFittedError):
        regular_data_processor.transform(regular_data_example)

def test_column_validations(regular_data_example, overlapped_column_lists, incomplete_column_lists):
    "Tests the column lists validation method."
    processor = RegularDataProcessor
    with raises(AssertionError):
        processor(*overlapped_column_lists).fit(regular_data_example)
    with raises(AssertionError):
        processor(*incomplete_column_lists).fit(regular_data_example)

def test_fit(regular_data_processor, regular_data_example):
    "Tests fit method and _check_is_fitted method before and after fitting."
    with raises(NotFittedError):
        regular_data_processor._check_is_fitted()
    processor = regular_data_processor.fit(regular_data_example)
    assert processor._check_is_fitted() is None

def test_fit_transform(regular_data_processor, regular_data_example):
    "Tests fit transform method, _check_is_fitted method and storing of attributes required for inverse_transform."
    transformed = regular_data_processor.fit_transform(regular_data_example)
    assert regular_data_processor._check_is_fitted() is None
    assert transformed.shape[0] == regular_data_example.shape[0]
    assert transformed.shape[1] != regular_data_example.shape[1]
    assert all([isinstance(idx, int) for idx in [regular_data_processor._num_col_idx_, regular_data_processor._cat_col_idx_]])
    assert isinstance(transformed, ndarray)

def test_inverse_transform(regular_data_processor, regular_data_example):
    "Tests inverse_transform and its output by comparing to the original data example."
    transformed = regular_data_processor.fit_transform(regular_data_example)
    inverted = regular_data_processor.inverse_transform(transformed)
    assert isinstance(inverted, type(regular_data_example))
    assert inverted.shape == regular_data_example.shape
    assert (inverted.columns == regular_data_example.columns).all()
    assert (inverted.dtypes == regular_data_processor._types).all()
    assert isclose(inverted, regular_data_example).all()
