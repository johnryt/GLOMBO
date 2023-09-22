# from https://github.com/garrettj403/SciencePlots

# to setup on Pypi: https://towardsdatascience.com/how-to-upload-your-python-package-to-pypi-de1b363a1b3

"""Install SciencePlots.
This script (setup.py) will copy the matplotlib styles (*.mplstyle) into the
appropriate directory on your computer (OS dependent).
This code is based on a StackOverflow answer:
https://stackoverflow.com/questions/31559225/how-to-ship-or-distribute-a-matplotlib-stylesheet
"""

import os
from subprocess import run

from setuptools import setup

# Get description from README
root = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(root, "readme.md"), "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="GLOMBO",
    version="0.0.1",
    packages=['GLOMBO'],
    author="John Ryter",
    author_email="jryter@usgs.gov",
    description="Run GLOMBO model, create figures, tables, and data associated with publication: Understanding key mineral supply chain dynamics using economics-informed material flow analysis and Bayesian optimization ",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    keywords=[
        "economics-informed material flow analysis",
        "material flow analysis",
        "MFA",
        "Bayesian optimization"
    ],
    url="https://github.com/johnryt/GLOMBO",
    include_package_data=True,
)

run(["conda", "env", "create", "-f", "glombo.yml"])