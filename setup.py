from setuptools import find_packages, setup

setup(
    name="phd-spc",
    version="1.0",
    description="Python package with selected SPC functions",
    author="Henrik Hviid Hansen",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "matplotlib",
        "numpy",
        "pandas",
        "scikit_learn",
        "scipy",
        "statsmodels",
    ],
)
