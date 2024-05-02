"""
Test suite for the RegularProcessor.
"""

import numpy as np  # isclose, ndarray
import pytest  # fixture, raises
from pmlb import fetch_data 
from sklearn.exceptions import NotFittedError

from ydata_synthetic.preprocessing.regular.processor import RegularDataProcessor

This initial block imports necessary libraries and modules. The test suite focuses on the RegularProcessor class from the ydata_synthetic library.

@fixture
def regular_data_example():
    return fetch_data('adult')

@fixture
def regular_data_processor_args(regular_data_example):
    num_cols = ['fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week']
    cat_cols = list(set(regular_data_example.columns).difference(set(num_cols)))
    return num_cols, cat_cols

These two fixtures create a synthetic dataset and the column lists for the RegularDataProcessor.

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

These two fixtures create column lists with overlapping and incomplete columns to test the column validation method.

@fixture
def regular_data_processor(regular_data_processor_args):
    num_cols, cat_cols = regular_data_processor_args
    return RegularDataProcessor(num_cols=num_cols, cat_cols=cat_cols)

This fixture creates a RegularDataProcessor instance with the given column lists.

def test_is_fitted(regular_data_processor, regular_data_example):
    "Tests raising NotFittedError in attempting to transform with a non fitted processor."
    This test checks if the transform method raises a NotFittedError when the processor is not fitted.

def test_column_validations(regular_data_example, overlapped_column_lists, incomplete_column_lists):
    "Tests the column lists validation method."
    This test checks if the validation method raises an AssertionError when the column lists overlap or are incomplete.

def test_fit(regular_data_processor, regular_data_example):
    "Tests fit method and _check_is_fitted method before and after fitting."
    This test checks if the fit method initializes the processor and if the _check_is_fitted method returns None after fitting.

def test_fit_transform(regular_data_processor, regular_data_example):
    "Tests fit_transform method, _check_is_fitted method and storing of attributes required for inverse_transform."
    This test checks the fit_transform method's output, the _check_is_fitted method after fitting, and if the processor stores necessary attributes for inverse_transform.

def test_inverse_transform(regular_data_processor, regular_data_example):
    "Tests inverse_transform and its output by comparing to the original data example."
    This test checks the inverse_transform method's output by comparing it to the original data example.
