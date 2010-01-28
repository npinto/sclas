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

DEFAULT_KERNEL_TYPE = "dot"
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

VALID_KERNEL_TYPES = ["dot", 
                      "ndot",
                      "exp_mu_chi2", 
                      "exp_mu_da"]



widgets = [RotatingMarker(), " Progress: ", Percentage(), " ",
           Bar(left='[',right=']'), ' ', ETA()]

# ------------------------------------------------------------------------------
def preprocess_features(features,
                        kernel_type = DEFAULT_KERNEL_TYPE,
                        whiten_vectors = None):
    
    assert(kernel_type in VALID_KERNEL_TYPES)

    features.shape = features.shape[0], -1

    if whiten_vectors is not None:
        fmean, fstd = whiten_vectors
        features -= fmean        
        assert((fstd!=0).all())
        features /= fstd

    if kernel_type == "exp_mu_chi2":
        fdiv = features.sum(1)[:,None]
        fdiv[fdiv==0] = 1
        return features / fdiv
    
    return features
    
# ------------------------------------------------------------------------------
def chi2_fromfeatures(features1,
                      features2 = None):

    if features2 is None:
        features2 = features1

    # set up progress bar        
    nfeat1 = len(features1)
    nfeat2 = len(features2)
    niter = nfeat1 * nfeat2
    pbar = ProgressBar(widgets=widgets, maxval=niter)
    pbar.start()

    # go
    n = 0
    kernelmatrix = sp.empty((nfeat1, nfeat2), dtype="float32")

    if features1 is features2:
        for ifeat1, feat1 in enumerate(features1):
            for ifeat2, feat2 in enumerate(features2):
                if ifeat1 == ifeat2:
                    kernelmatrix[ifeat1, ifeat2] = 0
                elif ifeat1 > ifeat2:
                    chi2dist = ne.evaluate("(((feat1 - feat2) ** 2.) / (feat1 + feat2) )")
                    chi2dist[sp.isnan(chi2dist)] = 0
                    chi2dist = chi2dist.sum()
                    kernelmatrix[ifeat1, ifeat2] = chi2dist
                    kernelmatrix[ifeat2, ifeat1] = chi2dist
                pbar.update(n+1)
                n += 1
    else:
        for ifeat1, feat1 in enumerate(features1):
            for ifeat2, feat2 in enumerate(features2):
                chi2dist = ne.evaluate("(((feat1 - feat2) ** 2.) / (feat1 + feat2) )")
                chi2dist[sp.isnan(chi2dist)] = 0
                chi2dist = chi2dist.sum()
                kernelmatrix[ifeat1, ifeat2] = chi2dist
                pbar.update(n+1)
                n += 1

    pbar.finish()    
    print "-"*80

    return kernelmatrix

# ------------------------------------------------------------------------------
def da_fromfeatures(features1,
                    features2 = None):

    if features2 is None:
        features2 = features1
        
    nfeat1 = len(features1)
    nfeat2 = len(features2)

    # go
    kernelmatrix = sp.empty((nfeat1, nfeat2), dtype="float32")

    if features1 is features2:

        # set up progress bar        
        n = 0
        niter = (nfeat1 * (nfeat2+1)) / 2
        pbar = ProgressBar(widgets=widgets, maxval=niter)
        pbar.start()

        for ifeat1, feat1 in enumerate(features1):

            # XXX: this is a hack that will only work with geometric blur d=204
            feat1 = feat1.reshape(-1, 204).copy()                    
            a2 = (feat1**2.).sum(1)[:,None]
        
            for ifeat2, feat2 in enumerate(features2):
                
                if ifeat1 == ifeat2:
                    kernelmatrix[ifeat1, ifeat2] = 0

                elif ifeat1 > ifeat2:
                    # XXX: this is a hack that will only work with geometric blur d=204
                    feat2 = feat2.reshape(-1, 204).copy()


                    ab = sp.dot(feat1, feat2.T)
                    
                    b2 = (feat2**2.).sum(1)[None,:]
                    res = (a2 - 2 *ab + b2)
            
                    dist = res.min(0).mean() + res.min(1).mean()

                    kernelmatrix[ifeat1, ifeat2] = dist
                    kernelmatrix[ifeat2, ifeat1] = dist
                    
                    pbar.update(n+1)
                    n += 1
    else:

        # set up progress bar        
        n = 0
        niter = nfeat1 * nfeat2
        pbar = ProgressBar(widgets=widgets, maxval=niter)
        pbar.start()

        for ifeat1, feat1 in enumerate(features1):

            # XXX: this is a hack that will only work with geometric blur d=204
            feat1 = feat1.reshape(-1, 204).copy()                    
            a2 = (feat1**2.).sum(1)[:,None]
        
            for ifeat2, feat2 in enumerate(features2):
                
                # XXX: this is a hack that will only work with geometric blur d=204
                feat2 = feat2.reshape(-1, 204).copy()


                ab = sp.dot(feat1, feat2.T)
                    
                b2 = (feat2**2.).sum(1)[None,:]
                res = (a2 - 2 *ab + b2)
                    
                dist = res.min(0).mean() + res.min(1).mean()
                
                kernelmatrix[ifeat1, ifeat2] = dist
                    
                pbar.update(n+1)
                n += 1        

    pbar.finish()
    print "-"*80

    return kernelmatrix

# ------------------------------------------------------------------------------
def dot_fromfeatures(features1,
                     features2 = None):

    if features2 is None:
        features2 = features1

    npoints1 = features1.shape[0]
    npoints2 = features2.shape[0]

    features1.shape = npoints1, -1
    features2.shape = npoints2, -1

    ndims = features1.shape[1]
    assert(features2.shape[1] == ndims)

    if ndims < DOT_MAX_NDIMS:
        out = sp.dot(features1, features2.T)
    else:
        out = sp.dot(features1[:,:DOT_MAX_NDIMS], 
                     features2[:,:DOT_MAX_NDIMS].T)
        ndims_done = DOT_MAX_NDIMS            
        while ndims_done < ndims:
            out += sp.dot(features1[:,ndims_done:ndims_done+DOT_MAX_NDIMS], 
                          features2[:,ndims_done:ndims_done+DOT_MAX_NDIMS].T)
            ndims_done += DOT_MAX_NDIMS
            
    return out

# ------------------------------------------------------------------------------
def ndot_fromfeatures(features1,
                     features2 = None):

    features1.shape = features1.shape[0], -1
    features1 = features1/sp.sqrt((features1**2.).sum(1))[:,None]

    if features2 is None:
        features2 = features1
    else:
        features2.shape = features2.shape[0], -1
        features2 = features2/sp.sqrt((features2**2.).sum(1))[:,None]

    return sp.dot(features1, features2.T)

# ------------------------------------------------------------------------------
def kernel_generate_fromcsv(input_csv_fname,
                            input_suffix,
                            output_fname,
                            kernel_type = DEFAULT_KERNEL_TYPE,
                            nowhiten = DEFAULT_NOWHITEN,
                            variable_name = DEFAULT_VARIABLE_NAME,
                            input_path = DEFAULT_INPUT_PATH,
                            overwrite = DEFAULT_OVERWRITE,
                            ):

    assert(kernel_type in VALID_KERNEL_TYPES)

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
    train_fnames = [ path.join(input_path, fname+input_suffix) 
                     for fname in ori_train_fnames ][:LIMIT]
    train_labels = [ row[1] for row in rows if row[2] == "train" ][:LIMIT]
    
    ori_test_fnames = [ row[0] for row in rows if row[2] == "test" ][:LIMIT]
    test_fnames = [ path.join(input_path, fname+input_suffix) 
                    for fname in ori_test_fnames ][:LIMIT]
    test_labels = [ row[1] for row in rows if row[2] == "test" ][:LIMIT]

    ntrain = len(train_fnames)
    ntest = len(test_fnames)

    # --------------------------------------------------------------------------
    # -- load features from train filenames
    # set up progress bar
    print "Loading training data ..."
    pbar = ProgressBar(widgets=widgets, maxval=ntrain)
    pbar.start()

    # load first vector to get dimensionality
    if kernel_type == "exp_mu_da":
        # hack for GB with 204 dims
        fvector0 = io.loadmat(train_fnames[0])[variable_name].reshape(-1, 204)
    else:
        fvector0 = io.loadmat(train_fnames[0])[variable_name].ravel()
    featshape = fvector0.shape
    featsize = fvector0.size

    # go
    train_features = sp.empty((ntrain,) + featshape, dtype='float32')
    error = False    
    for i, fname in enumerate(train_fnames):
        try:
            if kernel_type == "exp_mu_da":
                # hack for GB with 204 dims
                fvector = io.loadmat(fname)[variable_name].reshape(-1, 204)
            else:
                fvector = io.loadmat(fname)[variable_name].ravel()

        except TypeError:
            print "[ERROR] couldn't open", fname, "deleting it!"
            os.unlink(fname)
            error = True

        except:
            print "[ERROR] unkwon with", fname
            raise
        
        # XXX: revise that
        #fvector[sp.isnan(fvector)] = 0
        #fvector[sp.isinf(fvector)] = 0
        assert(not sp.isnan(fvector).any())
        assert(not sp.isinf(fvector).any())
        #if fvector.shape != fvector0.shape:
        #    print fname, fvector.shape, fvector0.shape
        #try:
        train_features[i] = fvector.reshape(fvector0.shape)
        #except:
            
        pbar.update(i+1)

    pbar.finish()
    print "-"*80

    if error:
        raise RuntimeError("An error occured (load train). Exiting.")        
        
    # -- train x train
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
     
            #fmean = train_features[:MEAN_MAX_NPOINTS].sum(0)
            #npoints_done = MEAN_MAX_NPOINTS
            #while npoints_done < npoints:
            #    fmean += train_features[npoints_done:npoints_done+MEAN_MAX_NPOINTS].sum(0)
            #    npoints_done += MEAN_MAX_NPOINTS

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

            # slow version:
            #fstd = ((train_features[:MEAN_MAX_NPOINTS]-fmean)**2.).sum(0)
            #npoints_done = MEAN_MAX_NPOINTS
            #while npoints_done < npoints:
            #    fstd += ((train_features[npoints_done:npoints_done+MEAN_MAX_NPOINTS]-fmean)**2.).sum(0)
            #    npoints_done += MEAN_MAX_NPOINTS

            fstd = sp.sqrt(fstd/npoints)

        fstd[fstd==0] = 1
        whiten_vectors = (fmean, fstd)
        train_features.shape = fshape
    train_features = preprocess_features(train_features, 
                                         kernel_type = kernel_type,
                                         whiten_vectors = whiten_vectors)
    assert(not sp.isnan(sp.ravel(train_features)).any())
    assert(not sp.isinf(sp.ravel(train_features)).any())

    print "Computing '%s' kernel_traintrain ..." % (kernel_type)
    if kernel_type == "dot":
        kernel_traintrain = dot_fromfeatures(train_features)
    elif kernel_type == "ndot":
        kernel_traintrain = ndot_fromfeatures(train_features)
    elif kernel_type == "exp_mu_chi2":
        chi2_matrix = chi2_fromfeatures(train_features)
        chi2_mu_train = chi2_matrix.mean()
        kernel_traintrain = ne.evaluate("exp(-chi2_matrix/chi2_mu_train)")        
    elif kernel_type == "exp_mu_da":
        da_matrix = da_fromfeatures(train_features)
        da_mu_train = da_matrix.mean()
        kernel_traintrain = ne.evaluate("exp(-da_matrix/da_mu_train)")        

    # --------------------------------------------------------------------------
    # -- load features from test filenames
    # set up progress bar
    print "Loading testing data ..."
    pbar = ProgressBar(widgets=widgets, maxval=ntest)
    pbar.start()

    # go
    test_features = sp.empty((ntest,) + featshape, dtype='float32')
    for i, fname in enumerate(test_fnames):

        try:
            if kernel_type == "exp_mu_da":            
                # hack for GB with 204 dims
                fvector = io.loadmat(fname)[variable_name].reshape(-1, 204)
            else:        
                fvector = io.loadmat(fname)[variable_name].ravel()
        except TypeError:
            print "[ERROR] couldn't open", fname, "deleting it"
            os.unlink(fname)
            error = True
        except:
            print "[ERROR] unkwon with", fname
            raise
            
        # XXX: revise that
        assert(not sp.isnan(fvector).any())
        assert(not sp.isinf(fvector).any())
        #if fvector.shape != fvector0.shape:
        #    print fname, fvector.shape, fvector0.shape
        test_features[i] = fvector.reshape(fvector0.shape)
        pbar.update(i+1)

    pbar.finish()
    print "-"*80

    if error:
        raise RuntimeError("An error occured (load test). Exiting.")        

    # -- train x test
    print "Preprocessing test features ..."
    test_features = preprocess_features(test_features, 
                                        kernel_type = kernel_type,
                                        whiten_vectors = whiten_vectors)
    assert(not sp.isnan(test_features).any())
    assert(not sp.isinf(test_features).any())

    print "Computing '%s' kernel_traintest ..."  % (kernel_type)
    if kernel_type == "dot":
        kernel_traintest = dot_fromfeatures(train_features, test_features)
    elif kernel_type == "ndot":
        kernel_traintest = ndot_fromfeatures(train_features, test_features)
    elif kernel_type == "exp_mu_chi2":
        chi2_matrix = chi2_fromfeatures(train_features, test_features)
        kernel_traintest = ne.evaluate("exp(-chi2_matrix/chi2_mu_train)")        
    elif kernel_type == "exp_mu_da":
        da_matrix = da_fromfeatures(train_features, test_features)
        kernel_traintest = ne.evaluate("exp(-da_matrix/da_mu_train)")        

    # --------------------------------------------------------------------------
    # -- write output file
    print
    print "Writing %s ..." % (output_fname)
    data = {"kernel_traintrain": kernel_traintrain,
            "kernel_traintest": kernel_traintest,
            "train_labels": train_labels,
            "test_labels": test_labels,
            "train_fnames": ori_train_fnames,
            "test_fnames": ori_test_fnames,
            }

    io.savemat(output_fname, data, format="4")

# ------------------------------------------------------------------------------
def main():

    usage = "usage: %prog [options] <input_csv_filename> <input_suffix> <output_filename>"
    
    parser = optparse.OptionParser(usage=usage)

    parser.add_option("--kernel_type", "-k",
                      type="str",                      
                      metavar="STR",
                      default=DEFAULT_KERNEL_TYPE,
                      help="'dot', 'exp_mu_chi2', 'exp_mu_da' [default='%default']")
    # TODO: 'cosine', 'exp_mu_intersect', 'intersect', 'chi2'

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

#     parser.add_option("--verbose", "-v" ,
#                       default=DEFAULT_VERBOSE,
#                       action="store_true",
#                       help="[default=%default]")

    opts, args = parser.parse_args()

    if len(args) != 3:
        parser.print_help()
    else:
        input_csv_fname = args[0]
        input_suffix = args[1]
        output_fname = args[2]

#         global verbose
#         verbose = opts.verbose
        
        kernel_generate_fromcsv(input_csv_fname,
                                input_suffix,
                                output_fname,
                                kernel_type = opts.kernel_type,
                                nowhiten = opts.nowhiten,
                                variable_name = opts.variable_name,
                                input_path = opts.input_path,
                                overwrite = opts.overwrite,
                                )

# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
