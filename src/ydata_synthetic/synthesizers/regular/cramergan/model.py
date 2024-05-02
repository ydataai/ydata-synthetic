"""
    CramerGAN model file
"""

import os
from os import path
from typing import List, Optional, NamedTuple

import numpy as np

import tensorflow as tf
from keras import  Model
from keras.layers import (Dense, Dropout, Input)
from keras.optimizers import Adam
from tqdm import trange

#Import ydata synthetic classes
from ....synthesizers import TrainParameters
from ....synthesizers.base import BaseGANModel
from ....synthesizers.loss import Mode, gradient_penalty


