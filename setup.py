#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name='argsrank',
    version='1.0.0',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        'flask',
    ],
)