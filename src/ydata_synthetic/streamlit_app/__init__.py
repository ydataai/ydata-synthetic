"""
    YData synthetic streamlit app for data synthesis
"""
from warnings import warn

from ydata_synthetic.streamlit_app import run

warn(
    "`import ydata_synthetic.streamllit_app` is deprecated. Please use **YData Fabric** instead."
    "For more information check https://docs.fabric.ydata.ai/latest/. To start today go to http://ydata.ai/register.",
    DeprecationWarning,
    stacklevel=2,
)