"""
    ydata_synthetic.synthesizers.regular init file
"""
from warnings import warn

warn(
    "`import ydata_synthetic.synthesizers.timeseries` is deprecated. Please use `import ydata.sdk.synthesizers import TimeSeriesSynthesizer` instead."
    "For more information check https://docs.synthetic.ydata.ai/latest and https://docs.fabric.ydata.ai/latest/sdk",
    DeprecationWarning,
    stacklevel=2,
)