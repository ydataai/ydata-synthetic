# Import required libraries
from fileinput import FileInput
from setuptools import setup, find_namespace_packages
from pathlib import Path

# Define the current working directory
here = Path(__file__).parent.resolve()

# Read package requirements from a text file
requirements = (here / "requirements.txt").read_text(encoding="utf8")

# Read the long description from a markdown file
long_description = (here / 'README.md').read_text(encoding='utf-8')

# Read the version number from a text file
version = (here / 'VERSION').read_text().rstrip("\n")

# Write the version number to the package source code
with open('src/ydata_synthetic/version.py', 'w') as version_file:
  version_file.write(f'__version__ = \'{version}\'')

# Configure the package setup
setup(
  # Package metadata
  name='ydata-synthetic',
  version=version,
  description='Synthetic data generation methods with different synthetization methods.',
  long_description=long_description,
  long_description_content_type='text/markdown',
  author='YData',
  author_email='community@ydata.ai',

  # Classifiers
  classifiers=[
    # Development status
    'Development Status :: 5 - Production/Stable',
    # Intended audience
    'Intended Audience :: Developers',
    'Intended Audience :: Education',
    'Intended Audience :: End Users/Desktop',
    'Intended Audience :: Financial and Insurance Industry',
    'Intended Audience :: Healthcare Industry',
    'Intended Audience :: Science/Research',
    'Intended Audience :: Telecommunications Industry',
    # License
    'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
    # Operating System
    'Operating System :: POSIX :: Linux',
    'Operating System :: MacOS',
    'Operating System :: Microsoft :: Windows',
    # Programming Language
    'Programming Language :: Python :: 3 :: Only',
    'Programming Language :: Python :: Implementation',
    # Topics
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Topic :: Software Development',
    'Topic :: Software Development :: Libraries :: Python Modules'
  ],

  # Keywords
  keywords='data science ydata',

  # URL and license
  url='https://github.com/ydataai/ydata-synthetic',
  license="https://github.com/ydataai/ydata-synthetic/blob/master/LICENSE",

  # Python version requirements
  python_requires=">=3.9, <3.12",

  # Package information
  packages=find_namespace_packages('src'),
  package_dir={'':'src'},
  include_package_data=True,
  options={"bdist_wheel": {"universal": True}},

  # Dependencies
  install_requires=requirements,

  # Extra dependencies
  extras_require={
    "streamlit": [
      "streamlit==1.29.0",
      "typing-extensions>=3.10.0",
      "streamlit_pandas_profiling==0.1.3",
      "ydata-profiling<5",
      "ydata-sdk>=0.2.1",
    ],
  },
)
