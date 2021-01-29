from setuptools import setup, find_namespace_packages
from pathlib import Path

here = Path(__file__).parent.resolve()

requirements = (here / "requirements.txt").read_text(encoding="utf8")
long_description = (here / 'README.md').read_text(encoding='utf-8')

version = (here / 'VERSION').read_text().rstrip("\n")

setup(name='ydata-synthetic',
      version=version,
      description='Synthetic data generation methods with different synthetization methods.',
      author='YData',
      author_email='community@ydata.ai',
      classifiers=[
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Software Development :: Artificial Intelligence :: Python Modules :: ',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',
      ],
      keywords='data science ydata',
      url='https://github.com/ydataai/ydata-synthetic',
      license="https://github.com/ydataai/ydata-synthetic/blob/master/LICENSE",
      python_requires=">=3.6, <3.9",
      packages=find_namespace_packages('src'),
      package_dir={'':'src'},
      include_package_data=True,
      options={"bdist_wheel": {"universal": True}},
      install_requires=requirements)
