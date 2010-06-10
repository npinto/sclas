#!/usr/bin/env python
# -*- coding: utf-8 -*-

# TODO: serious refactoring!

# TODO: clean + pylint
# TODO: chi2-mpi fast
# TODO: multiproc
# TODO: kernel_testtest ?
# TODO: cProfile / line_profiler / kernprof.py / cython?
# DEPENDENCY on numexpr!

# ------------------------------------------------------------------------------

import sys
import os
from os import path
import shutil
import optparse
import csv

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

# TODO: only use numexpr when available (no dependency)
try:
    import numexpr as ne
except ImportError:
    print("**Warning**: numexpr (ne) is missing. "
          "Code may raise exceptions!\n\n")

from utils import csv2tt, verify_fnames
from default import *

LIMIT = None
verbose = DEFAULT_VERBOSE

VALID_KERNEL_TYPES = ["dot", 
                      "ndot",
                      "exp_mu_chi2", 
                      "exp_mu_da",
                      ]


from fvector import get_fvector, VALID_SIMFUNCS

from pbar import *


# ------------------------------------------------------------------------------
from features import get_features, get_sphere_vectors, sphere_features
from kernel import dot_kernel, ndot_kernel, chi2_kernel, da_kernel

def kernel_generate_fromcsv(
    input_path,
    input_csv_fname,
    output_fname,
    # --
    featfunc,
    # -- 
    simfunc = DEFAULT_SIMFUNC,
    kernel_type = DEFAULT_KERNEL_TYPE,
    nosphere = DEFAULT_NOSPHERE,
    # --
    variable_name = DEFAULT_VARIABLE_NAME,
    #input_path = DEFAULT_INPUT_PATH,
    # --
    overwrite = DEFAULT_OVERWRITE,
    noverify = DEFAULT_NOVERIFY,
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
    
    (train_fnames, train_labels,
     test_fnames, test_labels) = csv2tt(input_csv_fname, input_path=input_path)
    
    ntrain = len(train_fnames)
    ntest = len(test_fnames)

    assert(ntrain>0)
    assert(ntest>0)

    if not noverify:
        all_fnames = sp.array(train_fnames+test_fnames).ravel()        
        verify_fnames(all_fnames)

    # --------------------------------------------------------------------------
    # -- train x train
    train_features = get_features(train_fnames,
                                  featfunc,
                                  kernel_type,
                                  simfunc,
                                  info_str = 'training')
    if nosphere:
        sphere_vectors = None
    else:
        print "Sphering train features ..."
        sphere_vectors = get_sphere_vectors(train_features)    
        train_features = sphere_features(train_features, sphere_vectors)

    # XXX: this should probably be refactored in kernel.py
    print "Computing '%s' kernel_traintrain ..." % (kernel_type)    
    if kernel_type == "dot":
        kernel_traintrain = dot_kernel(train_features)
    elif kernel_type == "ndot":
        kernel_traintrain = ndot_kernel(train_features)
    elif kernel_type == "exp_mu_chi2":
        chi2_matrix = chi2_kernel(train_features)
        chi2_mu_train = chi2_matrix.mean()
        kernel_traintrain = ne.evaluate("exp(-chi2_matrix/chi2_mu_train)")        
    elif kernel_type == "exp_mu_da":
        da_matrix = da_kernel(train_features)
        da_mu_train = da_matrix.mean()
        kernel_traintrain = ne.evaluate("exp(-da_matrix/da_mu_train)")        
    assert(not (kernel_traintrain==0).all())
    

    # --------------------------------------------------------------------------
    # -- train x test
    test_features = get_features(test_fnames,
                                 featfunc,
                                 kernel_type,
                                 simfunc,
                                 info_str = 'testing')
  
    if not nosphere:
        print "Sphering test features ..."
        test_features = sphere_features(test_features, sphere_vectors)
                
    # XXX: this should probably be refactored in kernel.py
    print "Computing '%s' kernel_traintest ..."  % (kernel_type)
    if kernel_type == "dot":
        kernel_traintest = dot_kernel(train_features, test_features)
    elif kernel_type == "ndot":
        kernel_traintest = ndot_kernel(train_features, test_features)
    elif kernel_type == "exp_mu_chi2":
        chi2_matrix = chi2_kernel(train_features, test_features)
        kernel_traintest = ne.evaluate("exp(-chi2_matrix/chi2_mu_train)")        
    elif kernel_type == "exp_mu_da":
        da_matrix = da_kernel(train_features, test_features)
        kernel_traintest = ne.evaluate("exp(-da_matrix/da_mu_train)")        

    assert(not (kernel_traintest==0).all())
    
    # --------------------------------------------------------------------------
    # -- write output file

    # first make sure we don't record the original input_path
    # since this one could change
    train_fnames, _, test_fnames, _ = csv2tt(input_csv_fname)
    
    print
    print "Writing %s ..." % (output_fname)
    
    data = {"kernel_traintrain": kernel_traintrain,
            "kernel_traintest": kernel_traintest,
            "train_labels": train_labels,
            "test_labels": test_labels,
            "train_fnames": train_fnames,
            "test_fnames": test_fnames,
            }

    try:
        io.savemat(output_fname, data, format="4")
    except IOError, err:
        print "ERROR!:", err
        

# ------------------------------------------------------------------------------
def get_optparser():
    
    usage = "usage: %prog [options] <input_csv_filename> <input_suffix> <output_filename>"
    
    parser = optparse.OptionParser(usage=usage)
    
    help_str = ("the similarity function to use "
                "(only for 'same/different' protocols) "
                "from the following list: %s. "
                % VALID_SIMFUNCS)
    parser.add_option("--simfunc", "-s",
                      type="choice",                      
                      #metavar="STR",
                      choices=VALID_SIMFUNCS,
                      default=DEFAULT_SIMFUNC,
                      help=help_str+"[DEFAULT='%default']"                      
                      )

    parser.add_option("--kernel_type", "-k",
                      type="str",                      
                      metavar="STR",
                      default=DEFAULT_KERNEL_TYPE,
                      help="'dot', 'exp_mu_chi2', 'exp_mu_da' [DEFAULT='%default']")
    # TODO: 'cosine', 'exp_mu_intersect', 'intersect', 'chi2' ?

    parser.add_option("--nosphere",
                      default=DEFAULT_NOSPHERE,
                      action="store_true",
                      help="[DEFAULT=%default]")

    parser.add_option("--variable_name", "-n",
                      metavar="STR",
                      type="str",
                      default=DEFAULT_VARIABLE_NAME,
                      help="[DEFAULT='%default']")

    parser.add_option("--input_path", "-i",
                      default=DEFAULT_INPUT_PATH,
                      type="str",
                      metavar="STR",
                      help="[DEFAULT='%default']")
    
    parser.add_option("--overwrite",
                      default=DEFAULT_OVERWRITE,
                      action="store_true",
                      help="overwrite existing file [default=%default]")

    parser.add_option("--noverify",
                      default=DEFAULT_NOVERIFY,
                      action="store_true",
                      help="disable verification of files before loading [default=%default]")

#     parser.add_option("--verbose", "-v" ,
#                       default=DEFAULT_VERBOSE,
#                       action="store_true",
#                       help="[default=%default]")
    # --
    
    
    return parser

        
# ------------------------------------------------------------------------------
def matfile_featfunc(fname,
                     suffix,
                     kernel_type = DEFAULT_KERNEL_TYPE,
                     variable_name = DEFAULT_VARIABLE_NAME):
    
    fname += suffix
    
    error = False
    try:
        if kernel_type == "exp_mu_da":
            # hack for GB with 204 dims
            fdata = io.loadmat(fname)[variable_name].reshape(-1, 204)
        else:
            fdata = io.loadmat(fname)[variable_name].ravel()

    except TypeError:
        fname_error = fname+'.error'
        print "[ERROR] couldn't open", fname, "moving it to", fname_error
        shutil.move(fname, fname_error)
        error = True

    except:
        print "[ERROR] (unknown) with", fname
        raise

    if error:
        raise RuntimeError("An error occured while loading '%s'"
                           % fname)

    assert(not sp.isnan(fdata).any())
    assert(not sp.isinf(fdata).any())

    return fdata
    
# ------------------------------------------------------------------------------
def kernel_generate_fromcsv_main(
    input_csv_fname,
    output_fname,
    # --
    input_suffix,
    input_path,
    variable_name = DEFAULT_VARIABLE_NAME,
    # -- 
    simfunc = DEFAULT_SIMFUNC,
    kernel_type = DEFAULT_KERNEL_TYPE,
    nosphere = DEFAULT_NOSPHERE,
    # --
    overwrite = DEFAULT_OVERWRITE,
    noverify = DEFAULT_NOVERIFY,
    ):

    def featfunc(fname):
        return matfile_featfunc(fname, input_suffix, kernel_type, variable_name)
    
    kernel_generate_fromcsv(
        input_path,
        input_csv_fname,
        output_fname,
        # --
        featfunc,
        # -- 
        simfunc = simfunc, 
        kernel_type = kernel_type,
        nosphere = nosphere,
        # --
        variable_name = variable_name,
        # --
        overwrite = overwrite,
        noverify = noverify
        )


def main():

    parser = get_optparser()
    
    opts, args = parser.parse_args()

    if len(args) != 3:
        parser.print_help()
    else:
        input_csv_fname = args[0]
        input_suffix = args[1]
        output_fname = args[2]

        kernel_generate_fromcsv_main(input_csv_fname,
                                     output_fname,
                                     # --
                                     get_vector_obj,
                                     # -- 
                                     simfunc = opts.simfunc, 
                                     kernel_type = opts.kernel_type,
                                     nosphere = opts.nosphere,
                                     # --
                                     variable_name = opts.variable_name,
                                     # --
                                     overwrite = opts.overwrite,
                                     noverify = opts.noverify
                                     )

    



# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
