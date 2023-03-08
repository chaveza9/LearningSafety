#!/usr/bin/env python

from setuptools import setup, find_packages

requirements = [
    "casadi",
    "matplotlib",
    "numpy",
    "seaborn",
    "tqdm",
    "torch",
    "cvxpylayers",
]

dev_requirements = [
    "black",
    "mypy",
    "pytest",
    "flake8",
]

setup(
    name="LearningBasedSafety",
    version="0.0.0",
    description="Safe control of dynamical systems using learning-based methods.",
    author="Andres Chavez Armijos",
    author_email="aschavez@bu.edu",
    url="",
    install_requires=requirements,
    extras_require={"dev": dev_requirements},
    packages=find_packages(),
)
