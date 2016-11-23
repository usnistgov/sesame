#!/usr/bin/env python
 
import sys
from setuptools import setup, Extension
from distutils.command.build_ext import build_ext
from distutils.errors import DistutilsPlatformError, DistutilsExecError, CCompilerError


ext_modules = [Extension('mumps._dmumps', sources=['mumps/_dmumps.c'])]

ext_errors = (CCompilerError, DistutilsExecError, DistutilsPlatformError)


class BuildFailed(Exception):
    def __init__(self):
        self.cause = sys.exc_info()[1]  # work around py 2/3 different syntax


class ve_build_ext(build_ext):
    # This class allows C extension building to fail.

    def run(self):
        try:
            build_ext.run(self)
        except DistutilsPlatformError:
            raise BuildFailed()

    def build_extension(self, ext):
        try:
            build_ext.build_extension(self, ext)
        except ext_errors:
            raise BuildFailed()
        except ValueError:
            # this can happen on Windows 64 bit, see Python issue 7511
            if "'path'" in str(sys.exc_info()[1]): # works with both py 2/3
                raise BuildFailed()
            raise

cmdclass = {'build_ext': ve_build_ext}



def status_msgs(*msgs):
    print('*' * 80)
    for msg in msgs:
        print(msg)
    print('*' * 80)



def run_setup(with_mumps):
    if with_mumps:
        kwargs = dict(ext_modules = ext_modules)
    else:
        kwargs = dict(ext_modules = [])

    setup(
        name = 'sesame',
        version = '0.1',
        author = 'Benoit H. Gaury',
        author_email = 'benoit.gaury@nist.gov',
        packages = ['sesame'],
        cmdclass = cmdclass,
        classifiers = [
            'Intended Audience :: Science/Research',
            'Programming Language :: Python :: 3',
        ],
        **kwargs
    )

try:
    run_setup(True)

    status_msgs(
        "BUILD SUMMARY: The MUMPS extension compiled successfully. Python build succeeded.")

except BuildFailed as exc:
    status_msgs(
        exc.cause,
        "WARNING: The MUMPS extension could not be compiled. " +
        "Retrying the build without the MUMPS extension now."
    )

    run_setup(False)

    status_msgs(
        "BUILD SUMMARY: The MUMPS extension could not be compiled. " +  
        "Plain-Python build succeeded."
    )
