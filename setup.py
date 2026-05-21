"""Installation script for the custom locomotion package."""

from setuptools import setup, find_packages

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    "mjlab==1.2.0",
    "mujoco-warp==3.5.0",
]

# Installation operation
setup(
    name="custom_locomotion_mjlab",
    packages=find_packages(include=("src", "src.*")),
    version="0.0.1",
    install_requires=INSTALL_REQUIRES,
)
