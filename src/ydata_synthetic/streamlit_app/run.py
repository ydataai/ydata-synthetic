"""
    Logic to run streamlit app from python code
"""
import os
from streamlit import config as _config
from streamlit.web import bootstrap

def run():
    dir_path = os.path.dirname(__file__)
    file_path = os.path.join(dir_path, "About.py")

    _config.set_option("server.headless", True)
    args = []

    bootstrap.run(file_path,'',args, flag_options={})