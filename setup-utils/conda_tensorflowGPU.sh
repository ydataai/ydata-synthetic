# Prompt the user to create a virtual environment
echo 'Create virtualenv? Type y or n followed by [ENTER]:'
read boolenv

# If the user wants to create a virtual environment
if [ $boolenv = "y" ]; then
  # Prompt the user for the virtual environment name
  echo "Provide virtual env name, followed by [ENTER]:"
  read envname
  
  # Create the virtual environment using conda
  conda create --name "$envname" --yes python=3.8

  # Print a message indicating that the virtual environment is being activated
  echo "Activating the created conda env"
  CONDA_BASE=$(conda info --base)
  source $CONDA_BASE/etc/profile.d/conda.sh
  conda activate "$envname"

# If the user doesn't want to create a virtual environment
else
  echo 'Creation of a new virtualenv is recommended to avoid potential conflicts'
fi

# Add NVIDIA package repositories
echo 'Adding NVIDIA package repositories'
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
sudo apt-get update

# Install NVIDIA machine learning repository
wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
sudo apt install -y ./nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
sudo apt-get update

# Install NVIDIA drivers and CUDA toolkit
echo 'Drivers installation'
sudo apt-get install --no-install-recommends nvidia-driver-450

# Install TensorRT dependencies
wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/libnvinfer7_7.1.3-1+cuda11.0_amd64.deb
sudo apt install ./libnvinfer7_7.1.3-1+cuda11.0_amd64.deb
sudo apt-get update

# Install CUDA and cuDNN
sudo apt-get install --no-install-recommends \
    cuda-11-0 \
    libcudnn8=8.0.4.30-1+cuda11.0  \
    libcudnn8-dev=8.0.4.30-1+cuda11.0

# Install TensorRT
echo 'Installing TensorRT. Requires that libcudnn8 is installed above.'
sudo apt-get install -y --no-install-recommends libnvinfer7=7.1.3-1+cuda11.0 \
    libnvinfer-dev=7.1.3-1+cuda11.0 \
    libnvinfer-plugin7=7.1.3-1+cuda11.0

# Upgrade pip and install ydata-synthetic package
echo 'Pip upgrade.'
pip3 install --upgrade pip
echo 'Installing ydata-synthetic package'
pip3 install ydata-synthetic

# Verify the success of the installation
echo 'Verifying the success of the installation'
python -c "import tensorflow as tf; print(tf.test.gpu_device_name())"
