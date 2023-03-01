from fileinput import FileInput
from setuptools import setup, find_namespace_packages
from pathlib import Path


here = Path(__file__).parent.resolve()

requirements = (here / "requirements.txt").read_text(encoding="utf8")
long_description = (here / 'README.md').read_text(encoding='utf-8')

version = (here / 'VERSION').read_text().rstrip("\n")

with open('src/ydata_synthetic/version.py', 'w') as version_file:
  version_file.write(f'__version__ = \'{version}\'')

setup(name='ydata-synthetic',
      version=version,
      description='Synthetic data generation methods with different synthetization methods.',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='YData',
      author_email='community@ydata.ai',
      classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: Financial and Insurance Industry',
        'Intended Audience :: Healthcare Industry',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Telecommunications Industry',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: Implementation',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries :: Python Modules'
      ],
      keywords='data science ydata',
      url='https://github.com/ydataai/ydata-synthetic',
      license="https://github.com/ydataai/ydata-synthetic/blob/master/LICENSE",
      python_requires=">=3.8, <3.11",
      packages=find_namespace_packages('src'),
      package_dir={'':'src'},
      include_package_data=True,
      options={"bdist_wheel": {"universal": True}},
      install_requires=requirements,
      extras_require={
          "streamlit": [
              "streamlit==0.18.1",
              "typing-extensions==3.10.0",
              "streamlit_pandas_profiling==0.1.3",
              "ydata-profiling==4.0.0"
          ],
      },
      )
