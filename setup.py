from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='phd-spc-methods',
    version='0.1.0',
    description='Package with SPC functions',
    author='Henrik Hviid Hansen',
    author_email='hehha@orsted.dk',
    classifiers=["Programming Language :: Python :: 3", "Operating System :: OS Independent", ],
    packages=find_packages(),
)

