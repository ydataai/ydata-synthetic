"""
    Main time-series synthesizer class
"""
from enum import Enum, unique
import os
from joblib import load

from tensorflow import config as tfconfig

