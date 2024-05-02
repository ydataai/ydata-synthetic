"""
    Dataset cache utility functions
    --------------------------------
    
    This module contains utility functions for caching datasets used by the
    pandas-profiling package. These functions help in checking if a dataset
    is already available in the cache, and if not, downloading and saving it.
    
    The original code can be found at 
    https://github.com/ydataai/pandas-profiling/blob/master/src/pandas_profiling/utils/
"""

import os
import zipfile
from pathlib import Path

import requests

def get_project_root() -> Path:
    """Returns the path to the project root folder.
    
    Returns:
        The path to the project root folder.
    """
    # The Path class from the pathlib module is used to handle file paths.
    # Here, it is used to get the parent directory of the current file,
    # and then getting the parent directory of that directory,
    # which should be the project root folder.
    return Path(__file__).parent.parent.parent.parent

def get_data_path() -> Path:
    """Returns the path to the dataset cache ([root] / data)
    
    Returns:
        The path to the dataset cache
    """
    # The get_project_root() function is used to get the project root folder,
    # and then the 'data' directory is created inside it if it doesn't already exist.
    data_path = get_project_root() / "data"
    data_path.mkdir(exist_ok=True)
    return data_path

def cache_file(file_name: str, url: str) -> Path:
    """Check if file_name already is in the data path, otherwise download it from url.
    
    Args:
        file_name: the file name
        url: the URL of the dataset
    
    Returns:
        The relative path to the dataset
    """
    # The get_data_path() function is used to get the path to the dataset cache.
    data_path = get_data_path()

    # The file_path is the path to the dataset file inside the dataset cache.
    file_path = data_path / file_name

    # If the file_path does not exist, it is created by downloading the dataset
    # from the provided URL using the requests library.
    if not file_path.exists():
        response = requests.get(url)
        file_path.write_bytes(response.content)

    # The file_path is returned as the relative path to the dataset.
    return file_path

def cache_zipped_file(file_name: str, url: str) -> Path:
    """Check if file_name already is in the data path, otherwise download it from url.
    
    Args:
        file_name: the file name
        url: the URL of the zipped dataset
    
    Returns:
        The relative path to the dataset
    """
    # The get_data_path() function is used to get the path to the dataset cache.
    data_path = get_data_path()

    # The file_path is the path to the dataset file inside the dataset cache.
    file_path = data_path / file_name

    # If the file_path does not exist, it is created by downloading the zipped dataset
    # from the provided URL using the requests library.
    if not file_path.exists():
        response = requests.get(url)

        # If the response status code is not 200 (OK), a FileNotFoundError is raised.
        if response.status_code != 200:
            raise FileNotFoundError("Could not download resource")

        # A temporary file 'tmp.zip' is created to store the downloaded zipped dataset.
        tmp_path = data_path / "tmp.zip"

        # The downloaded zipped dataset is written to the temporary file.
        tmp_path.write_bytes(response.content)

        # The zipped dataset is extracted to the dataset cache using the zipfile library.
        with zipfile.ZipFile(tmp_path, "r") as zip_file:
            zip_file.extract(file_path.name, data_path)

        # The temporary file is deleted after the zipped dataset is extracted.
        tmp_path.unlink()

    # The file_path is returned as the relative path to the dataset.
    return file_path
