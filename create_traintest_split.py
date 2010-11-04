#!/usr/bin/env python
# -*- coding: utf-8 -*-

import optparse
import csv
from numpy.random import shuffle, seed
from os.path import isdir, basename, splitext, exists
import warnings
from glob import glob
from pprint import pprint
import scipy as sp

DEFAULT_NTRAIN = "2"
DEFAULT_NTEST = "2"
DEFAULT_OVERWRITE = False
DEFAULT_RSEED = None

EXTENSIONS = [".jpg", ".bmp", ".png", ".pgm", ".tif", ".tiff"]

DEBUG = False

# ------------------------------------------------------------------------------
def create_traintest_split(input_path,
                           output_fname,
                           ntrain = DEFAULT_NTRAIN,
                           ntest = DEFAULT_NTEST,
                           overwrite = DEFAULT_OVERWRITE,
                           rseed = DEFAULT_RSEED,
                           ):

    # can we overwrite ?
    if exists(output_fname) and not overwrite:
        warnings.warn("not allowed to overwrite %s"  % output_fname)
        return

    # -- get categories
    cat_paths = [ fil 
                  for fil in glob(input_path + "/*") 
                  if isdir(fil) ]

    cats = [ basename(cat_path) 
             for cat_path in cat_paths ]

    # -- get filenames per category
    cat_fnames = {}
    for cat, cat_path in zip(cats, cat_paths):

        # get filenames with an image extension only
        cfn = [ cat + "/" + basename(fname) 
                for fname in glob(cat_path + "/*") 
                if splitext(fname)[-1].lower() in EXTENSIONS ]

        # verify that there is enough
        if len(cfn) < ntrain+ntest:
            print len(cfn), ntrain+ntest
            raise ValueError, "there is not enough images"
        
        # shuffle and get what you need
        seed(rseed)
        shuffle(cfn)
        cfn = cfn[:ntrain+ntest]

        # split
        cfn_train = cfn[:ntrain]
        cfn_test = cfn[ntrain:ntrain+ntest]

        # save
        cat_fnames[cat] = {"train": cfn_train, "test": cfn_test}
    
    # -- generate train/test lists
    train_list = []
    for cat in cat_fnames:
        for fname in cat_fnames[cat]["train"]:
            train_list += [ (fname, cat, "train") ]

    test_list = []
    for cat in cat_fnames:
        for fname in cat_fnames[cat]["test"]:
            test_list += [ (fname, cat, "test") ]

    assert len(cats) != 0
    assert len(train_list) != 0
    assert len(test_list) != 0

    # -- write csv file
    print "="*80
    print "number of categories:", len(cats)
    print sp.array(cats)
    print "-"*80
    print "number of training examples:", len(train_list)
    print "number of testing examples:", len(test_list)
    print "-"*80
    print "output filename:", output_fname
    csvw = csv.writer(open(output_fname, "w+"))
    csvw.writerows(train_list)
    csvw.writerows(test_list)

    if DEBUG:
        del csvw
        res = [e for e in csv.reader(open(output_fname)) ]
        pprint(res)
    

# ------------------------------------------------------------------------------
def main():

    usage = "usage: %prog [options] <input_path> <output_filename>"

    parser = optparse.OptionParser(usage=usage)

    parser.add_option("--ntrain", 
                      default=DEFAULT_NTRAIN,
                      type="str",
                      metavar="INT",
                      help="[default=%default]")

    parser.add_option("--ntest", 
                      default=DEFAULT_NTEST,
                      type="str",
                      metavar="INT",
                      help="[default=%default]")

    parser.add_option("--overwrite", 
                      default=DEFAULT_OVERWRITE,
                      action="store_true",
                      help="overwrite existing file [default=%default]")

    parser.add_option("--rseed",
                      type="str",
                      default=DEFAULT_RSEED,
                      metavar="INT",
                      help="fix the random seed (if not None) [default=%default]")

    opts, args = parser.parse_args()

    if len(args) != 2:
        parser.print_help()
    else:
        input_path = args[0]
        output_fname = args[1]

        create_traintest_split(input_path,
                               output_fname,
                               # --
                               ntrain = int(opts.ntrain),
                               ntest = int(opts.ntest),
                               # --
                               overwrite = opts.overwrite,
                               rseed = int(opts.rseed),
                               )        


# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
