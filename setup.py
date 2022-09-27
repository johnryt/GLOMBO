# from https://github.com/johnryt/generalization

# to setup on Pypi: https://towardsdatascience.com/how-to-upload-your-python-package-to-pypi-de1b363a1b3

"""
Install generalization.
This script (setup.py) will ...

"""

import os

from setuptools import setup, find_packages
from setuptools.command.install import install


# Get description from README
root = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(root, "readme.md"), "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="generalization",
    version="0.0.1",
    packages=find_packages(exclude=["tests.*", "tests", "figs", "examples"]),
    author="John Ryter",
    author_email="ryterj@mit.edu",
    description="all the code for the generalization model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    keywords=[
        "Generalization",
        "Criticality",
        "Displacement",
    ],
    url="https://github.com/johnryt/generalization",
    install_requires=[
        "os"
    ],
    include_package_data=True,
)