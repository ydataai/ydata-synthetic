"""
    Logic to run streamlit app from python code
"""
from warnings import warn

def run():
    warn(
        "`import ydata_synthetic.streamllit_app` is deprecated. Please use **YData Fabric** instead."
        "For more information check https://docs.fabric.ydata.ai/latest/. To start today go to http://ydata.ai/register.",
        DeprecationWarning,
        stacklevel=2,
    )