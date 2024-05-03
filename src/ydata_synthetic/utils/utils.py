"""
    Utility functions that are common to ydata-synthetic project
"""
import os
import subprocess
import platform
import requests

from ydata_synthetic.version import __version__
def analytics_features(model: str):
    endpoint= "https://packages.ydata.ai/ydata-synthetic?"

    if bool(os.getenv("YDATA_SYNTHETIC_NO_ANALYTICS"))!= True:
        package_version = __version__
        try:
            subprocess.check_output("nvidia-smi")
            gpu_present = True
        except Exception:
            gpu_present = False

        python_version = ".".join(platform.python_version().split(".")[:2])

        try:
            request_message = f"{endpoint}version={package_version}" \
                                 f"&python_version={python_version}" \
                                 f"&model={model}" \
                                 f"&os={platform.system()}" \
                                 f"&gpu={str(gpu_present)}"

            requests.get(request_message)
        except Exception:
            pass
