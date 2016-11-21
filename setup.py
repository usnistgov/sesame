#!/usr/bin/env python

from setuptools import setup, Extension

setup(
    name='sesame',
    version='0.1',
    author='Benoit H. Gaury',
    author_email='benoit.gaury@nist.gov',
    packages=['mumps', 'sesame'],
    ext_modules=[Extension('mumps._dmumps', sources=['mumps/_dmumps.c'], libraries=['dmumps'])],
    classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
    ]
)
