"""
    Logic to run Streamlit app from Python code

    This module contains the necessary logic to run a Streamlit app from a Python script. It imports the required
    modules and functions from Streamlit, and defines a `run` function that takes no arguments.

    The `run` function sets the Streamlit configuration option for running in headless mode (i.e., without a
    graphical user interface), and then constructs the file path for the main app script (in this case, "About.py").

    Finally, the `run` function calls the `bootstrap.run` function to launch the Streamlit app, passing in the file
    path, any additional arguments, and an empty dictionary of flag options.
"""
import os
from streamlit import config as _config
from streamlit.web import bootstrap

def run():
    """
        Run the Streamlit app

        This function sets the necessary configuration options and launches the Streamlit app. It first sets the
        `server.headless` option to True to disable the graphical user interface.

        It then constructs the file path for the main app script using the `os.path` module.

        Finally, it calls the `bootstrap.run` function to launch the Streamlit app, passing in the file path, any
        additional arguments, and an empty dictionary of flag options.
    """
    dir_path = os.path.dirname(__file__)
    file_path = os.path.join(dir_path, "About.py")

    _config.set_option("server.headless", True)
    args = []

    bootstrap.run(file_path, '', args, flag_options={})
