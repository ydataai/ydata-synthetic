from functools import partial
from joblib import dump
import numpy as np
from pandas import DataFrame
import tensorflow as tf
from keras.layers import \
    (Input, Dense, LeakyReLU, Dropout, BatchNormalization, ReLU, Concatenate)
from keras import Model


