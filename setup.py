#!/usr/bin/python3
from setuptools import find_packages, setup

setup(
    name='src',
    package_dir={'src': 'src', 'src.data': 'src/data', 'src.tools': 'src/tools'},
    packages=find_packages(),
    version='0.1.0',
    description='Concise description of the project.',
    author='Birgitte Nilsson',
    license='MIT')


find_packages()