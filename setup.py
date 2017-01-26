#from distutils.core import setup
import setuptools

setuptools.setup(
    name='BluePyEfe',
    version='0.1dev',
    packages=['bluepyefe'],
    install_requires=['igorpy'],
    dependency_links=['git+ssh://bbpcode.epfl.ch/user/vangeit/igorpy.git'],
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
)
