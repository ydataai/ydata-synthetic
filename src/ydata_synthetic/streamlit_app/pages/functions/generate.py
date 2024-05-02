import pandas as pd
from ydata_profiling import ProfileReport

try:
    from streamlit_pandas_profiling import st_profile_report
except ImportError:
    st_profile_report = lambda x: None

from ydata_synthetic.synthesizers.regular import RegularSynthesizer
from ydata_synthetic.synthesizers.timeseries import TimeGAN
from ydata_synthetic.streamlit_app.pages.functions.train import DataType

def load_model(input_path: str, datatype: DataType) -> any:
    """
    Load a synthetic data model from disk.

    Args:
        input_path (str): The path to the saved model.
        datatype (DataType): The type of the model to load.

    Returns:
        A synthetic data model.
    """
    try:
        if datatype == DataType.TABULAR:
            model = RegularSynthesizer.load(input_path)
        else:
            model = TimeGAN.load(input_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    return model

def generate_profile(df: pd.DataFrame):
    """
    Generate a data profile report for a given DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to profile.
    """
    report = ProfileReport(df, title='Synthetic data profile', interactions=None)
    if st_profile_report:
        st_profile_report(report)
