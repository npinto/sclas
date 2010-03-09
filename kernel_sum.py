#!/usr/bin/python
# -*- coding: utf-8 -*-

import optparse
import sys
from os import path
import warnings

import scipy as sp
import numpy as np
from scipy import io

# ------------------------------------------------------------------------------
DEFAULT_WEIGHTS = []
DEFAULT_NO_TRACE_NORMALIZATION = False
DEFAULT_OVERWRITE = False

# ------------------------------------------------------------------------------
def kernel_sum(output_filename,
               input_filenames,
               weights = DEFAULT_WEIGHTS,
               no_trace_normalization = DEFAULT_NO_TRACE_NORMALIZATION,               
               overwrite = DEFAULT_OVERWRITE,
               ):

    """ TODO: docstring """
    
    assert len(weights) <= len(input_filenames)

    # can we overwrite ?
    if path.exists(output_filename) and not overwrite:
        warnings.warn("not allowed to overwrite %s"  % output_filename)
        return

    # --        
    lw = len(weights)
    lf = len(input_filenames)
    if lw <= lf:
        weights += [1. for _ in xrange(lf-lw)]
    print "Using weights =", weights

    # -- check


    # -- 
    kernel_train = None
    kernel_test = None
    train_fnames = None
    test_fnames = None
    print "Loading %d file(s) ..."  % lf
    for ifn, (fname, weight) in enumerate(zip(input_filenames, weights)):

        print "%d %s (weight=%f)"  % (ifn+1, fname, weight)

        if weight == 0:
            continue

        kernel_mat = io.loadmat(fname)


        # -- check that kernels come from the same "source"
        if train_fnames is None:
            train_fnames = kernel_mat["train_fnames"]
            test_fnames = kernel_mat["test_fnames"]
        else:
            #print train_fnames, kernel_mat["train_fnames"]
            assert (train_fnames == kernel_mat["train_fnames"]).all()
            assert (test_fnames == kernel_mat["test_fnames"]).all()

        ktrn = kernel_mat['kernel_traintrain']
        ktst = kernel_mat['kernel_traintest']

        assert(not (ktrn==0).all() )
        assert(not (ktst==0).all() )

        if not no_trace_normalization:
            ktrn_trace = ktrn.trace()
            ktrn /= ktrn_trace
            ktst /= ktrn_trace

        if kernel_train is None:
            kernel_train = weight * ktrn
            kernel_test = weight * ktst
        else:
            kernel_train += weight * ktrn
            kernel_test += weight * ktst

    train_labels = sp.array([str(elt) for elt in kernel_mat["train_labels"]])
    test_labels = sp.array([str(elt) for elt in kernel_mat["test_labels"]])

    if not no_trace_normalization:
        kernel_train_trace = kernel_train.trace()
        kernel_train /= kernel_train_trace
        kernel_test /= kernel_train_trace

    kernel_train = kernel_train.astype('float64')
    kernel_test = kernel_test.astype('float64')

    # --------------------------------------------------------------------------
    # -- write output file
    print "Writing %s ..." % (output_filename)
    data = {"kernel_traintrain": kernel_train,
            "kernel_traintest": kernel_test,
            "train_labels": train_labels,
            "test_labels": test_labels,
            "train_fnames": [str(fname) for fname in train_fnames],
            "test_fnames": [str(fname) for fname in test_fnames],
            "input_filenames": input_filenames,
            "weights": weights,
            }

    io.savemat(output_filename, data, format="4")

# ------------------------------------------------------------------------------
def main():

    """ TODO: docstring """    

    usage = "usage: %prog [options] <output_filename> <input_filename1> <...>"
    
    parser = optparse.OptionParser(usage=usage)
    
    # -- 
    parser.add_option("--weight", "-w",
                      type = "float",
                      dest = "weights",
                      action = "append",
                      default = DEFAULT_WEIGHTS,
                      help = "[default=%default]")

    parser.add_option("--overwrite",
                      default=DEFAULT_OVERWRITE,
                      action="store_true",
                      help="overwrite existing file [default=%default]")

    parser.add_option("--no_trace_normalization", 
                      action="store_true", 
                      default=DEFAULT_NO_TRACE_NORMALIZATION,
                      help="[default=%default]")

    opts, args = parser.parse_args()

    if len(args) < 2:
        parser.print_help()
    else:
        output_filename = args[0]
        input_filenames = args[1:]
            
        kernel_sum(output_filename,
                   input_filenames,                   
                   weights = opts.weights,
                   overwrite = opts.overwrite,
                   )
                       
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()






