import os
import numpy

def makedir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def makedirs(filename): # also accepts filename
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

def print_dict(v, prefix=''):
    if isinstance(v, dict):
        for k, v2 in v.items():
            p2 = "{}['{}']".format(prefix, k)
            print_dict(v2, p2)
    elif isinstance(v, list):
        for i, v2 in enumerate(v):
            p2 = "{}[{}]".format(prefix, i)
            print_dict(v2, p2)
    else:
        print('{}: {} {}'.format(prefix, numpy.shape(v), type(v)))
