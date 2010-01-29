#!/usr/bin/env python

# TODO: optparse + cleaning

import warnings
warnings.simplefilter('ignore', FutureWarning)

import sys
from scipy import io

fname = sys.argv[1]
try:
    key = sys.argv[2]
except IndexError:
    key = None


matfile = io.loadmat(fname)

if key is None:
    print matfile
else:
    print matfile[key]


