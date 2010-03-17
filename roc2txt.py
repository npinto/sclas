#!/usr/bin/python
# -*- coding: utf-8 -*-

""" TODO: docstring """
#from future import __division__

import optparse
import sys
import os
from os import path
import warnings
import shutil
import itertools

import scipy as sp
from scipy import io
import numpy as np
import pylab as pl

# ------------------------------------------------------------------------------
DEFAULT_OVERWRITE = False
DEFAULT_SHOW = False

# ------------------------------------------------------------------------------
def roc2txt(output_filename,
            input_filenames,
            # --
            show = DEFAULT_SHOW,
            overwrite = DEFAULT_OVERWRITE,
            ):

    if path.exists(output_filename) and not overwrite:
        warnings.warn("not allowed to overwrite %s"  % output_filename)
        return

    final_test_fnames = sp.empty((0,2), dtype=str)
    final_test_y = []
    final_test_pred = []

    sets = set([])

    # -- extract data from svm outputs
    print "Collecting data ..."
    for fname in input_filenames:
        print fname
        data = io.loadmat(fname)

        # test_y
        test_y = data['test_y']
        # for now only accept binary classifiers
        if test_y.shape[1] != 2:
            raise NotImplementedError("test_y.shape[1] != 2, "
                                      "binary classifiers only!!!")        
        # test_fnames
        test_fnames = data['test_fnames']
        if len(set(test_fnames) & sets) != 0:
            raise ValueError("Duplicate input detected!")
        sets = sets or set(test_fnames)
        test_fnames.shape = len(test_y), -1
        # for now only accept same / different protocol
        assert test_fnames.shape[1] == 2        
        assert test_y.shape[0] == len(test_fnames)
        test_y = test_y[:,1]

        # test_predictions
        test_predictions = data['test_predictions']
        # assume same / different binary protocol
        assert test_predictions.shape[0] == 1
        test_predictions.shape = -1

        # flip sign?
        person1 = path.split(test_fnames[0,0])[0]
        person2 = path.split(test_fnames[0,1])[0]
        if person1 == person2:
            if test_y[0] > 0:
                flip_sign = 1.
            else:
                flip_sign = -1.
        elif person1 != person2:
            if test_y[0] > 0:
                flip_sign = -1.
            else:
                flip_sign = 1.

        final_test_fnames = sp.concatenate((final_test_fnames, test_fnames))        
        final_test_y = sp.concatenate((final_test_y, flip_sign*test_y))
        final_test_pred = sp.concatenate((final_test_pred, flip_sign*test_predictions))

        print final_test_fnames.shape, final_test_pred.shape
        
    # -- verify data (based on same/different protocol a-la-LFW)
    print "Verifying data ..."
    for y, fnames in zip(final_test_y, final_test_fnames):
        names = [path.split(fname)[0] for fname in fnames]
        if y > 0:
            assert names[0] == names[1]
        else:
            assert names[0] != names[1]
    
    # -- true and false positive rates
    pos = final_test_y>0
    neg = final_test_y<=0

    assert final_test_y.shape == final_test_pred.shape

    pmin, pmax = final_test_pred.min(), final_test_pred.max()
    print final_test_pred.shape
    print pmin, pmax

    thresholds = sp.unique(final_test_pred[final_test_pred.argsort()])

    tpr_l = []
    fpr_l = []
    for i, th in enumerate(thresholds[::-1]):
        tpr = (final_test_pred[pos]>th).mean()
        fpr = (final_test_pred[neg]>th).mean()
        # correct roc curve (make it convex)
        if i != 0 and tpr_l[i-1] > tpr: tpr = tpr_l[i-1]
        tpr_l += [tpr]
        fpr_l += [fpr]



    print "npoints=", len(tpr_l)
    print "Saving", output_filename
    sp.savetxt(output_filename, sp.array([tpr_l, fpr_l]).T, fmt='%1.10f')

    if show:
        #v1 = sp.loadtxt('funneled-v1-like-roc.txt')
        #pl.plot(v1[:,1], v1[:,0])
        #wolf = sp.loadtxt('accv09-wolf-hassner-taigman-roc.txt')
        #pl.plot(wolf[:,1], wolf[:,0])
        this = sp.loadtxt(output_filename)
        print this.shape
        pl.plot(this[:,1], this[:,0])
        pl.show()
    
    
# ------------------------------------------------------------------------------
def main():

    """ TODO: docstring """    

    usage = ("usage: %prog [options] "
             "<output_filename> "
             "<input_filename1> [<input_filename2>, ... ]")
    
    parser = optparse.OptionParser(usage=usage)
    
    parser.add_option("--overwrite",
                      default=DEFAULT_OVERWRITE,
                      action="store_true",
                      help="overwrite existing file [default=%default]")

    parser.add_option("--show", "-s",
                      default=DEFAULT_SHOW,
                      action="store_true",
                      help="plot the roc curve and show it [default=%default]")

    opts, args = parser.parse_args()

    if len(args) < 2:
        parser.print_help()
    else:

        output_filename = args[0]
        input_filenames = args[1:]

        roc2txt(output_filename,
                input_filenames,
                # --
                show = opts.show,
                overwrite = opts.overwrite,
                )
        
                       
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()






