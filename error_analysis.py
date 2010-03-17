#!/usr/bin/python
# -*- coding: utf-8 -*-

""" TODO: docstring """

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
import cPickle as pkl

# ------------------------------------------------------------------------------
DEFAULT_OVERWRITE = False
DEFAULT_INPUT_PATH = "./"

# ------------------------------------------------------------------------------
def error_analysis(output_dir,
                   input_fname1,
                   input_fname2,# = None,
                   # --
                   input_path = DEFAULT_INPUT_PATH,                     
                   overwrite = DEFAULT_OVERWRITE,
                   ):

    """ TODO: docstring """

    if path.exists(output_dir):
        if not overwrite:
            warnings.warn("not allowed to overwrite %s"  % output_dir)
            return
        else:
            shutil.rmtree(output_dir)
        
    os.makedirs(output_dir)

    print "Loading", input_fname1
    data1 = io.loadmat(input_fname1)
    print "Loading", input_fname2
    data2 = io.loadmat(input_fname2)

    d1 = {}
    d2 = {}

    for ddict, data in zip([d1, d2], [data1, data2]):
        test_labels = data['test_labels']
        test_y = data['test_y']
        svm_labels = data['svm_labels'].ravel()
        test_predictions = data['test_predictions'].ravel()
        accuracy = data['accuracy']
        test_fnames = data['test_fnames']    
    
        labels = sp.unique(test_labels)
        assert len(labels) == 2, "len(labels) > 2 not implemented"

        # WARNING: this code will only work for the same/different protocol (for now)
        #   Two types of svm outputs can be handled:
        #   . those with 'different' / 'same' (with potential sign flip)
        #   . those with the correct +1 / -1
        if list(labels) == ['different', 'same']:
            y2label = dict(sp.unique(zip(svm_labels, test_labels)))
            label2y = dict(sp.unique(zip(test_labels, svm_labels)))
            if float(label2y['same']) == -1:
                flip_sign = -1
            else:
                flip_sign = 1
            test_y_1d_idx = (sp.array([int(item) for item in svm_labels])[:,None] == test_y).sum(0) > 0
            y_gt = flip_sign * test_y[:, test_y_1d_idx].ravel()
            y_pred = sp.sign(flip_sign * test_predictions)
            assert abs(100.*(y_gt==y_pred).mean() - accuracy) < 1e-6
        elif list(labels) == [-1, 1]:
            y2label = dict(sp.unique(zip(svm_labels.ravel(), test_labels.ravel())))
            label2y = dict(sp.unique(zip(test_labels.ravel(), svm_labels.ravel())))
            if float(label2y[1]) == -1:
                flip_sign = -1
            else:
                flip_sign = 1
            test_y_1d_idx = (sp.array([int(item) for item in svm_labels])[:,None] == test_y).sum(0) > 0
            y_gt = flip_sign * test_y[:, test_y_1d_idx].ravel()
            y_pred = sp.sign(flip_sign * test_predictions)
            assert abs(100.*(y_gt==y_pred).mean() - accuracy) < 1e-6
            
        else:
            raise NotImplementedError

        ddict['y_gt'] = y_gt
        ddict['y_pred'] = y_pred
        ddict['test_fnames'] = test_fnames

        pos = y_gt > 0
        neg = y_gt <= 0

        assert pos.sum() + neg.sum() == y_pred.size

        assert y_gt.size == y_pred.size

        tp = y_pred[pos] > 0
        fn = y_pred[pos] <= 0
        assert tp.sum() + fn.sum() == pos.sum()

        tn = y_pred[neg] <= 0
        fp = y_pred[neg] > 0
        assert tn.sum() + fp.sum() == neg.sum()
        
        ddict['tp'] = (pos, tp)
        ddict['fp'] = (neg, fp)
        
        ddict['tn'] = (neg, tn)
        ddict['fn'] = (pos, fn)        

#         ddict['tp'] = (pos, sp.bitwise_and(y_pred > 0, pos))
#         ddict['fp'] = (neg, sp.bitwise_and(y_pred > 0, neg))
        
#         ddict['tn'] = (neg, sp.bitwise_and(y_pred <= 0, neg))
#         ddict['fn'] = (pos, sp.bitwise_and(y_pred <= 0, pos))

    assert (d1['y_gt'] == d2['y_gt']).all()

    print "total:", len(test_y)

    # -- compute hamming distance
    items = 'tp', 'fn', 'tn', 'fp'
    output_d = dict([(item, {}) for item in items])
    for item in items:
        x = d1[item][1]
        y = d2[item][1]

        assert x.size == y.size

        ham = (x!=y).sum()
        overlap = 100.*(1.-(1.*ham/x.size))
        print "%s hamming: %d, size: %d, percent overlap: %.2f" % (item, ham, x.size, overlap)
        output_d[item]['hamming'] = ham
        output_d[item]['size'] = x.size
        output_d[item]['overlap'] = overlap
        
    # -- output images
    assert (d1['test_fnames'] == d2['test_fnames']).all()
    fnames = [path.join(input_path, fname) for fname in d1['test_fnames']]

    nfnames = len(fnames)
    for n, fname in enumerate(fnames):
        assert path.exists(fname)
        sys.stdout.write("\rVerify that the files exist: "
                         "%6.2f%%" % (100.*(n+1.)/nfnames))
        sys.stdout.flush()
    print

    fnames = sp.array(fnames).reshape(len(fnames)/2, 2)


    pos_items = [(a,b) for a in 'tp','fn' for b in 'tp','fn']
    neg_items = [(a,b) for a in 'tn','fp' for b in 'tn','fp']

    all_items = pos_items + neg_items

    for xitem, yitem in all_items:
        
        idx, x = d1[xitem]
        idy, y = d2[yitem]
        assert (idx==idy).all()

        sel = sp.bitwise_and(x, y)
        mfnames = fnames[idx][sel]

        nmfnames = len(mfnames)

        if nmfnames <= 0:
            continue

        mpath = path.join(output_dir, xitem+'_'+yitem)            
        os.makedirs(mpath)
        for n, (fname1, fname2) in enumerate(mfnames):
            arr1 = sp.atleast_3d(sp.misc.imread(fname1).astype('float32'))
            arr2 = sp.atleast_3d(sp.misc.imread(fname2).astype('float32'))
            # grayscale?
            if arr1.shape[2] == 1:
                arr1 = sp.concatenate([arr1,arr1,arr1], 2)
            if arr2.shape[2] == 1:
                arr2 = sp.concatenate([arr2,arr2,arr2], 2)
            fullarr = sp.concatenate([arr1, arr2], axis=1)
            basename1 = path.basename(fname1)
            basename2 = path.basename(fname2)
            out_fname = path.join(mpath, "%s-%s" % (basename1, basename2))
            sp.misc.imsave(out_fname, fullarr)
            sys.stdout.write("\rProcessing '%s': "
                             "%6.2f%%" % (mpath, (100.*(n+1.)/nmfnames)))
            sys.stdout.flush()
            
        print

    out_fname = path.join(output_dir, "error_analysis.pkl")
    print "Saving", out_fname
    pkl.dump(output_d, open(out_fname, 'w+'), protocol=2)
    
# ------------------------------------------------------------------------------
def main():

    """ TODO: docstring """    

    usage = ("usage: %prog [options] "
             "<output_dir> "
             "<input_filename1> [<input_filename2>]")
    
    parser = optparse.OptionParser(usage=usage)
    
    parser.add_option("--input_path", "-i",
                      default=DEFAULT_INPUT_PATH,
                      type="str",
                      metavar="STR",
                      help="[DEFAULT='%default']")
    
    parser.add_option("--overwrite",
                      default=DEFAULT_OVERWRITE,
                      action="store_true",
                      help="overwrite existing file [default=%default]")

    opts, args = parser.parse_args()

    #if len(args) != 2 and len(args) != 3:
    if len(args) != 3:
        parser.print_help()
    else:

        output_dir = args[0]
        input_fname1 = args[1]
        input_fname2 = args[2]

#         if len(args) == 3:
#             input_fname2 = args[2]
#         else:
#             input_fname2 = None

        error_analysis(output_dir,
                       input_fname1,
                       input_fname2,# = input_fname2,
                       # --
                       input_path = opts.input_path,
                       overwrite = opts.overwrite,
                       )
                       
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()






