"""
    Auxiliary functions for synthetic data training
"""
from enum import Enum
import streamlit as st

from ydata_synthetic.synthesizers.regular import RegularSynthesizer
from ydata_synthetic.synthesizers.timeseries.timegan.model import TimeGAN
from ydata_synthetic.synthesizers import ModelParameters

__MODEL_MAPPING = {'tabular': RegularSynthesizer, 'timeseries': TimeGAN}
__CONDITIONAL_MODELS = ['CGAN', 'CWGANGP']

class DataType(Enum):
    TABULAR = 'tabular'
    TIMESERIES = 'timeseries'

def init_synth(datatype: DataType, modelname: str, model_parameters: ModelParameters, n_critic: int=1) -> any:
    if datatype not in __MODEL_MAPPING:
        raise ValueError(f"Invalid datatype: {datatype}. Valid datatypes are: {', '.join(map(str, DataType))}")
    
    synth = __MODEL_MAPPING[datatype]
    modelname = modelname.lower()
    
    if modelname not in synth.available_models:
        raise ValueError(f"Invalid model name: {modelname} for datatype: {datatype}")
    
    if modelname in ['wgan', 'cwgangp', 'wgangp']:
        if datatype != DataType.TABULAR:
            raise ValueError(f"Model {modelname} is not available for datatype: {datatype}")
        synth = synth(modelname=modelname,
                                   model_parameters=model_parameters,
                                   n_critic=n_critic)
    else:
        synth = synth(modelname=modelname,
                                   model_parameters=model_parameters)
    return synth

def advanced_settings() -> tuple[int, int, float, float]:
    col1, col2 = st.columns(2)
    with col1:
        noise_dim = st.number_input('Select noise dimension', 0, 200, 128, 1)
        layer_dim = st.number_input('Select the layer dimension', 0, 200, 128, 1)
    with col2:
        beta_1 = st.slider('Select first beta co-efficient', 0.0, 1.0, 0.5)
        beta_2 = st.slider('Select second beta co-efficient', 0.0, 1.0, 0.9)
    return noise_dim, layer_dim, beta_1, beta_2

def training_parameters(model_name:str, df_cols: list[str]) -> tuple[int, list[str] | None]:
    col1, col2 = st.columns([2, 4])
    with col1:
        epochs = st.number_input('Epochs', min_value=0, value=100)

    if model_name in __CONDITIONAL_MODELS:
        with col2:
            label_col = st.multiselect('Choose the conditional cols:', df_cols)
            if not label_col:
                st.warning("Please select at least one conditional column")
    else:
        label_col = None
    return epochs, label_col

