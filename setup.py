from setuptools import setup, find_packages
from setuptools.extension import Extension
import numpy as np
import sys

setup(
    name='intprim',
    version='2.0',
    description='Interaction Primitives library from the Interactive Robotics Lab at Arizona State University',
    url='https://github.com/ir-lab/intprim',
    author='Joseph Campbell',
    author_email='jacampb1@asu.edu',
    license='MIT',
    packages=find_packages()
)
