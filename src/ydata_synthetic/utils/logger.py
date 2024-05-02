"""
    ydata-synthetic logger
"""
from __future__ import absolute_import, division, print_function

import logging

import pandas as pd

from ydata_synthetic.utils.utils import analytics_features

class SynthesizersLogger(logging.Logger):
    def __init__(self, name, level=logging.INFO):
        super().__init__(name, level)

    def info(
            self,
            msg: object,
            moment: str = 'training'
        ) -> None:
        super().info(f'[SYNTHESIZER] {moment} - {msg}.')

    def info_def_report(self, model: str):
        analytics_features(model=model)

        super().info(f'[SYNTHESIZER] Creating a synthetic data generator with the following model - {model}.')