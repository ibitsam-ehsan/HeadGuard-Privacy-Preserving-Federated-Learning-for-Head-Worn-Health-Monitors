from setuptools import setup, find_packages

setup(
    name="headguard",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.19.0",
        "scipy>=1.5.0",
        "scikit-learn>=0.24.0",
        "pyyaml>=5.4.0",
    ],
)
