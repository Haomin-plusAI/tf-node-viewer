import os
from setuptools import setup

LONG_DESC = open('README.md').read()
LICENSE = open('LICENSE').read()

repository_dir = os.path.dirname(__file__)

with open(os.path.join(repository_dir, 'requirements.txt')) as fh:
        dependencies = fh.readlines()

setup(
    name="tf_node_viewer",
    version="0.0.1",
    description="A reverse engineering tool which provide the user an easy way to extract data from a graph",
    install_requires=dependencies,
    long_description=LONG_DESC,
    url='https://github.com/neil-tan/tf-node-viewer',
    author='Neil Tan',
    author_email='michael.bartling15@gmail.com',
    license=LICENSE,
    packages=["view_node"],
)
