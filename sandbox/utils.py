
import sys
from os import path
import csv
import scipy as sp

def csv2tt(input_fname, input_path=""):

    csvr = csv.reader(open(input_fname))
    rows = [ row for row in csvr ]
    
    train_fnames = [ [ path.join(input_path, fname)
                       for fname in row[:-2] ]
                     for row in rows
                     if row[-1] == "train" ]
    train_labels = [ row[-2]
                     for row in rows
                     if row[-1] == "train" ]
    
    test_fnames = [ [ path.join(input_path, fname)
                      for fname in row[:-2] ]
                    for row in rows
                    if row[-1] == "test" ]
    test_labels = [ row[-2]
                    for row in rows
                    if row[-1] == "test" ]

    return (train_fnames, train_labels,
            test_fnames, test_labels)

def csvglob2tt(input_glob, input_path=""):
    raise NotImplementedError

def path2tt(path, ntrain, ntest, ntrials):
    raise NotImplementedError

def verify_fnames(fnames):
    for n, fname in enumerate(fnames):
        sys.stdout.write("Verifying that the necessary files exist:"
                         " %02.2f%%\r" % (100.*(n+1)/fnames.size))
        sys.stdout.flush()
        if not path.exists(fname):
            raise IOError("File '%s' doesn't exist!" % fname)
        if path.getsize(fname)==0:
            raise IOError("File '%s' is empty!" % fname)
    
