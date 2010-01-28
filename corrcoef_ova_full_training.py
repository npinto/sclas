#!/usr/bin/env python
# -*- coding: utf-8 -*-

# TODO: clean + pylint
# TODO: chi2-mpi fast
# TODO: multiproc
# TODO: kernel_testtest ?
# TODO: cProfile / line_profiler / kernprof.py / cython?
# DEPENDENCY on numexpr!

# ------------------------------------------------------------------------------

import sys
import os
import os.path as path
import optparse
import csv

#from loadmat import loadmat

import warnings
warnings.simplefilter('ignore', FutureWarning)

try:
    import scipy as sp
    from scipy import (
        io, linalg,
        )
except ImportError:
    print "scipy is missing (sudo easy_install -U scipy)"
    raise

try:
    import numexpr as ne
except ImportError:
    print "numexpr is missing (sudo easy_install -U numexpr)"
    raise

from npprogressbar import *

# ------------------------------------------------------------------------------

DEFAULT_CORRCOEF_TYPE = 'pos_neg_mean'
DEFAULT_NOWHITEN = False
DEFAULT_VARIABLE_NAME = "data"
DEFAULT_INPUT_PATH = "./"
DEFAULT_OVERWRITE = False
DEFAULT_VERBOSE = False

LIMIT = None

verbose = DEFAULT_VERBOSE

DOT_MAX_NDIMS = 10000
MEAN_MAX_NPOINTS = 2000
STD_MAX_NPOINTS = 2000

widgets = [RotatingMarker(), " Progress: ", Percentage(), " ",
           Bar(left='[',right=']'), ' ', ETA()]

# ------------------------------------------------------------------------------
def preprocess_features(features,
                        whiten_vectors = None):
    
    features.shape = features.shape[0], -1

    if whiten_vectors is not None:
        fmean, fstd = whiten_vectors
        features -= fmean        
        assert((fstd!=0).all())
        features /= fstd

    return features
    
# ------------------------------------------------------------------------------
def kernel_generate_fromcsv(input_csv_fname,
                            input_suffix,
                            output_fname,
                            corrcoef_type = DEFAULT_CORRCOEF_TYPE,
                            nowhiten = DEFAULT_NOWHITEN,
                            variable_name = DEFAULT_VARIABLE_NAME,
                            input_path = DEFAULT_INPUT_PATH,
                            overwrite = DEFAULT_OVERWRITE,
                            ):

    # add matlab's extension to the output filename if needed
    if path.splitext(output_fname)[-1] != ".mat":
        output_fname += ".mat"        

    # can we overwrite ?
    if path.exists(output_fname) and not overwrite:
        warnings.warn("not allowed to overwrite %s"  % output_fname)
        return
        
    # --------------------------------------------------------------------------
    # -- get training and testing filenames from csv 
    print "Processing %s ..." % input_csv_fname
    csvr = csv.reader(open(input_csv_fname))
    rows = [ row for row in csvr ]
    ori_train_fnames = [ row[0] for row in rows if row[2] == "train" ][:LIMIT]
    train_fnames = sp.array([ path.join(input_path, fname+input_suffix) 
                     for fname in ori_train_fnames ][:LIMIT])
    train_labels = sp.array([ row[1] for row in rows if row[2] == "train" ][:LIMIT])
    
    ori_test_fnames = [ row[0] for row in rows if row[2] == "test" ][:LIMIT]
    test_fnames = sp.array([ path.join(input_path, fname+input_suffix) 
                    for fname in ori_test_fnames ][:LIMIT])
    test_labels = sp.array([ row[1] for row in rows if row[2] == "test" ][:LIMIT])

    ntrain = len(train_fnames)
    ntest = len(test_fnames)

    # --------------------------------------------------------------------------
    # -- load features from train filenames
    # set up progress bar
    print "Loading training data ..."
    pbar = ProgressBar(widgets=widgets, maxval=ntrain)
    pbar.start()

    fvector0 = io.loadmat(train_fnames[0])[variable_name].ravel()
    featshape = fvector0.shape
    featsize = fvector0.size

    # go
    train_features = sp.empty((ntrain,) + featshape, dtype='float32')
    error = False    
    for i, fname in enumerate(train_fnames):
        try:
            fvector = io.loadmat(fname)[variable_name].ravel()

        except TypeError:
            print "[ERROR] couldn't open", fname, "deleting it!"
            os.unlink(fname)
            error = True

        except:
            print "[ERROR] unkwon with", fname
            raise
        
        assert(not sp.isnan(fvector).any())
        assert(not sp.isinf(fvector).any())
        train_features[i] = fvector.reshape(fvector0.shape)
        
        pbar.update(i+1)

    pbar.finish()
    print "-"*80

    if error:
        raise RuntimeError("An error occured (load train). Exiting.")        
        
    # -- preprocess train
    print "Preprocessing train features ..."
    if nowhiten:
        whiten_vectors = None
    else:
        fshape = train_features.shape
        train_features.shape = fshape[0], -1
        npoints, ndims = train_features.shape

        if npoints < MEAN_MAX_NPOINTS:
            fmean = train_features.mean(0)
        else:
            # - try to optimize memory usage...
            sel = train_features[:MEAN_MAX_NPOINTS]
            fmean = sp.empty_like(sel[0,:])

            sp.add.reduce(sel, axis=0, dtype="float32", out=fmean)

            curr = sp.empty_like(fmean)
            npoints_done = MEAN_MAX_NPOINTS
            while npoints_done < npoints:
                sel = train_features[npoints_done:npoints_done+MEAN_MAX_NPOINTS]
                sp.add.reduce(sel, axis=0, dtype="float32", out=curr)
                sp.add(fmean, curr, fmean)
                npoints_done += MEAN_MAX_NPOINTS                
     
            fmean /= npoints

        if npoints < STD_MAX_NPOINTS:
            fstd = train_features.std(0)
        else:
            # - try to optimize memory usage...

            sel = train_features[:MEAN_MAX_NPOINTS]

            mem = sp.empty_like(sel)
            curr = sp.empty_like(mem[0,:])

            seln = sel.shape[0]
            sp.subtract(sel, fmean, mem[:seln])
            sp.multiply(mem[:seln], mem[:seln], mem[:seln])
            fstd = sp.add.reduce(mem[:seln], axis=0, dtype="float32")

            npoints_done = MEAN_MAX_NPOINTS
            while npoints_done < npoints:
                sel = train_features[npoints_done:npoints_done+MEAN_MAX_NPOINTS]
                seln = sel.shape[0]
                sp.subtract(sel, fmean, mem[:seln])
                sp.multiply(mem[:seln], mem[:seln], mem[:seln])
                sp.add.reduce(mem[:seln], axis=0, dtype="float32", out=curr)
                sp.add(fstd, curr, fstd)

                npoints_done += MEAN_MAX_NPOINTS

            fstd = sp.sqrt(fstd/npoints)

        fstd[fstd==0] = 1
        whiten_vectors = (fmean, fstd)
        train_features.shape = fshape
    train_features = preprocess_features(train_features, 
                                         whiten_vectors = whiten_vectors)
    assert(not sp.isnan(sp.ravel(train_features)).any())
    assert(not sp.isinf(sp.ravel(train_features)).any())

    # -- train
    categories = sp.unique(train_labels)
    #if categories.size == 2:
    #    categories = [categories[0]]
    #else:
    #    raise NotImplementedError("not sure if it works with ncats > 2")

    corrcoef_kernels = {}
    cat_index = {}
    for icat, cat in enumerate(categories):

        if corrcoef_type == 'pos_neg_mean':
            #print train_features.shape
            #print train_features[train_labels == cat].shape
            #print train_features[train_labels != cat].shape
            corrcoef_ker = train_features[train_labels == cat].sum(0) \
                           - train_features[train_labels != cat].sum(0)
            corrcoef_ker /= ntrain 
        elif corrcoef_type == 'pos_mean_neg_mean':
            corrcoef_ker = train_features[train_labels == cat].mean(0) \
                           - train_features[train_labels != cat].mean(0)
        elif corrcoef_type == 'pos_mean':
            corrcoef_ker = train_features[train_labels == cat].mean(0)
        else:
            raise ValueError("corrcoef_type '%s' not understood"
                             % corrcoef_type)

        corrcoef_ker -= corrcoef_ker.mean()
        corrcoef_ker_mag = sp.linalg.norm(corrcoef_ker)
        assert corrcoef_ker_mag > 0
        corrcoef_ker /= sp.linalg.norm(corrcoef_ker)

        assert(not sp.isnan(corrcoef_ker).any())
        assert(not sp.isinf(corrcoef_ker).any())
        
        corrcoef_kernels[cat] = corrcoef_ker
        cat_index[cat] = icat 

    # --------------------------------------------------------------------------
    # -- load features from test filenames
    # set up progress bar
    print "Testing (on the fly) ..."
    pbar = ProgressBar(widgets=widgets, maxval=ntest)
    pbar.start()

    # -- test
    # XXX: code adapted from beta svm_ova_fromfilenames (to review!)
    pred = sp.zeros((ntest))
    distances = sp.zeros((ntest, len(categories)))

    for itest, fname in enumerate(test_fnames):

        try:
            fvector = io.loadmat(fname)[variable_name].ravel()
        except TypeError:
            print "[ERROR] couldn't open", fname, "deleting it"
            os.unlink(fname)
            error = True
        except:
            print "[ERROR] unkwon with", fname
            raise
            
        assert(not sp.isnan(fvector).any())
        assert(not sp.isinf(fvector).any())

        # whiten if needed
        if whiten_vectors is not None:
            fmean, fstd = whiten_vectors
            fvector -= fmean        
            assert((fstd!=0).all())
            fvector /= fstd            
        
        assert(not sp.isnan(fvector).any())
        assert(not sp.isinf(fvector).any())

        # corrcoef
        testv = fvector
        testv -= testv.mean()
        testv_mag = sp.linalg.norm(testv)
        assert testv_mag > 0
        testv /= testv_mag

        for icat, cat in enumerate(categories):

            corrcoef_ker = corrcoef_kernels[cat]
            resp = sp.dot(testv, corrcoef_ker)

            distances[itest, icat] = resp        
        
        pbar.update(itest+1)

    pbar.finish()
    print "-"*80

    if error:
        raise RuntimeError("An error occured (load test). Exiting.")        

    if len(categories) > 1:
        pred = distances.argmax(1)
        #print sp.array([cat_index[e] for e in test_labels]).astype('int')
        gt = sp.array([cat_index[e] for e in test_labels]).astype("int")
        perf = (pred == gt)

        accuracy = 100.*perf.sum() / ntest
    else:
        pred = sp.sign(distances).ravel()
        gt = sp.array(test_labels)
        cat = categories[0]
        gt[gt != cat] = -1
        gt[gt == cat] = +1
        gt = gt.astype("int")
        perf = (pred == gt)
        accuracy = 100.*perf.sum() / ntest        
        
    print distances.shape
    print "Classification accuracy on test data (%):", accuracy


    svm_labels = gt

    # -- average precision
    # XXX: redo this part to handle other labels than +1 / -1
    ap = 0
    if distances.shape[1] == 1:
        distances = distances.ravel()
        assert test_labels.ndim == 1            
        assert svm_labels.ndim == 1
    
        # -- inverse predictions if needed
        # (that is when svm was trained with flipped +1/-1 labels)
        # -- convert test_labels to +1/-1 (int)
        try:
            test_labels = array([int(elt) for elt in test_labels])
            if (test_labels != svm_labels).any():
                distances = -distances

            #if not ((test_labels==-1).any() and (test_labels==1).any()):
            #    test_labels[test_labels!=test_labels[0]] = +1
            #    test_labels[test_labels==test_labels[0]] = -1

            #print test_labels

            # -- convert test_labels to +1/-1 (int)
            test_labels = array([int(elt) for elt in test_labels])

            # -- get average precision
            c = distances
            #print c
            si = sp.argsort(-c)
            tp = sp.cumsum(sp.single(test_labels[si]>0))
            fp = sp.cumsum(sp.single(test_labels[si]<0))
            rec  = tp/sp.sum(test_labels>0)
            prec = tp/(fp+tp)

            #print prec, rec
            #from pylab import *
            #plot(prec, rec)
            #show()

            ap = 0
            rng = sp.arange(0,1.1,.1)
            for th in rng:
                p = prec[rec>=th].max()
                if p == []:
                    p = 0
                ap += p / rng.size

            print "Average Precision:", ap

        except ValueError:
            ap = 0


    # XXX: clean this
    test_y = sp.array([svm_labels.ravel()==lab
                       for lab in sp.unique(svm_labels.ravel())]
                      )*2-1
    test_y = test_y.T

    print distances
    
    # --------------------------------------------------------------------------
    # -- write output file
    if output_fname is not None:
        print "Writing %s ..." % (output_fname)
        # TODO: save more stuff (alphas, etc.)
        data = {"accuracy": accuracy,
                "average_precision":ap,
                "test_distances": distances,
                "test_labels": test_labels,
                "test_y": test_y,
                "svm_labels": svm_labels,
                }

        io.savemat(output_fname, data, format='4')

    return accuracy    


# ------------------------------------------------------------------------------
def main():

    usage = "usage: %prog [options] <input_csv_filename> <input_suffix> <output_filename>"
    
    parser = optparse.OptionParser(usage=usage)

    parser.add_option("--corrcoef_type", "-t",
                      type="str",                      
                      metavar="STR",
                      default=DEFAULT_CORRCOEF_TYPE,
                      help="'pos_neg_mean', 'pos_mean_neg_mean', 'pos_mean' [default='%default']")

    parser.add_option("--nowhiten",
                      default=DEFAULT_NOWHITEN,
                      action="store_true",
                      help="[default=%default]")

    parser.add_option("--variable_name", "-n",
                      metavar="STR",
                      type="str",
                      default=DEFAULT_VARIABLE_NAME,
                      help="[default='%default']")

    parser.add_option("--input_path", "-i",
                      default=DEFAULT_INPUT_PATH,
                      type="str",
                      metavar="STR",
                      help="[default='%default']")
    
    parser.add_option("--overwrite",
                      default=DEFAULT_OVERWRITE,
                      action="store_true",
                      help="overwrite existing file [default=%default]")

    opts, args = parser.parse_args()

    if len(args) != 3:
        parser.print_help()
    else:
        input_csv_fname = args[0]
        input_suffix = args[1]
        output_fname = args[2]
        
        kernel_generate_fromcsv(input_csv_fname,
                                input_suffix,
                                output_fname,
                                corrcoef_type = opts.corrcoef_type,
                                nowhiten = opts.nowhiten,
                                variable_name = opts.variable_name,
                                input_path = opts.input_path,
                                overwrite = opts.overwrite,
                                )

# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
