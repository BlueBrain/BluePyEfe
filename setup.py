#from distutils.core import setup
import setuptools
import subprocess

setuptools.setup(
    name='BluePyEfe',
    version='0.1.1dev',
    description='bluepyefe',
    packages=setuptools.find_packages(),
    install_requires=['igorpy','matplotlib','efel','sh'],
    dependency_links=['git+ssh://bbpcode.epfl.ch/user/vangeit/igorpy.git#egg=igorpy'],
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
)
