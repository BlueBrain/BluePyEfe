import os
import numpy

def makedir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def makedirs(filename): # also accepts filename
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
