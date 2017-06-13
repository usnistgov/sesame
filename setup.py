#!/usr/bin/env python3

###############################################################################
#
# Author: Benoit Gaury <benoit.gaury@nist.gov>
# Author: Paul M. Haney <paul.haney@nist.gov>
#
# =============================================================================
#
# This file is part of Sesame.
#
# The Sesame software was developed at the National Institute of Standards and
# Technology (NIST) by employees of the
# Federal Government in the course of their official duties. Pursuant to title
# 17 Section 105 of the United States Code this software is not subject to
# copyright protection and is in the public domain.
#
# You may use, copy and distribute copies of the software in any medium,
# provided that you keep intact this entire notice. You may improve, modify and
# create derivative works of the software or any portion of the software, and
# you may copy and distribute such modifications or works. Modified works should
# carry a notice stating that you changed the software and should note the date
# and nature of any such change.  Please explicitly acknowledge the National
# Institute of Standards and Technology as the source of the software. 
#
# The software is expressly provided "AS IS". NIST MAKES NO WARRANTY OF ANY
# KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING,
# WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A
# PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER
# REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE
# UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES
# NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR
# THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY,
# RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
#
# You are solely responsible for determining the appropriateness of using and
# distributing the software and you assume all risks associated with its use,
# including but not limited to the risks and costs of program errors, compliance
# with applicable laws, damage to or loss of data, programs or equipment, and
# the unavailability or interruption of operation. This software is not intended
# to be used in any situation where a failure could cause risk of injury or
# damage to property. 
#
###############################################################################
 
import sys
import configparser
from setuptools import setup, Extension
from distutils.command.build_ext import build_ext
from distutils.errors import DistutilsPlatformError, DistutilsExecError, CCompilerError


CONFIG_FILE = 'setup.cfg'

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
    print()
    for msg in msgs:
        print(msg)
    print()



def run_setup(packages, ext_modules):
    # populate the version_info dictionary with values stored in the version file
    version_info = {}
    with open('sesame/_version.py', 'r') as f:
        exec(f.read(), {}, version_info)
    setup(
        name = 'sesame',
        version = version_info['__version__'],
        author = 'Benoit H. Gaury',
        author_email = 'benoit.gaury@nist.gov',
        packages = packages,
        cmdclass = cmdclass,
        ext_modules = ext_modules,
        classifiers = [
            'Intended Audience :: Science/Research',
            'Programming Language :: Python :: 3',
        ],
    )


config = configparser.ConfigParser()
try:
    with open(CONFIG_FILE) as f:
        config.readfp(f)
except IOError:
    print("Could not open config file.")


if 'mumps' in config.sections():
    kwrds = {}
    for name, value in config.items('mumps'):
        kwrds[name] = value

    ext_modules = [Extension(
        'sesame.mumps._dmumps',
        sources=['sesame/mumps/_dmumps.c'],  
        libraries=[kwrds['libraries']],
        library_dirs=[kwrds['library_dirs']],
        include_dirs=[kwrds['include_dirs']])]

    packages = ['sesame', 'sesame.mumps']

    try:
        run_setup(packages, ext_modules)
        status_msgs("Done")
    except BuildFailed as exc:
        status_msgs(
            exc.cause,
            "WARNING: The MUMPS extension could not be compiled. " +
            "Retrying the build without the MUMPS extension now.")
        run_setup(['sesame'], [])
        status_msgs("Done")
else:
    run_setup(['sesame'], [])
    status_msgs( "Done")
