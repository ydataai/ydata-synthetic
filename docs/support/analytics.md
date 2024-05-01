# Analytics & Telemetry

## Overview

`ydata-synthetic` is a powerful library designed to generate synthetic data. 
As part of our ongoing efforts to improve user experience and functionality, `ydata-synthetic` includes a telemetry feature.
This feature collects anonymous usage data, helping us understand how the library is used and identify areas for improvement.

The primary goal of collecting telemetry data is to:

- Enhance the functionality and performance of the ydata-synthetic library
- Prioritize new features based on user engagement
- Identify common issues and bugs to improve overall user experience

### Data Collected
The telemetry system collects non-personal, anonymous information such as:

- Python version
- `ydata-synthetic` version
- Frequency of use of `ydata-synthetic` features
- Errors or exceptions thrown within the library

## Disabling usage analytics

We respect your choice to not participate in our telemetry collection. If you prefer to disable telemetry, you can do so
by setting an environment variable on your system. Disabling telemetry will not affect the functionality of the ydata-profiling library,
except for the ability to contribute to its usage analytics.

### Set an Environment Variable:
In your notebook or script make sure to set YDATA_SYNTHETIC_NO_ANALYTICS environment variable to `True`.

````python
    import os
    
    os.environ['YDATA_SYNTHETIC_NO_ANALYTICS']=True
````




