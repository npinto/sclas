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

DEFAULT_NIMGS = 100
DEFAULT_NCV = 10
DEFAULT_RSEED = None

DEFAULT_OVERWRITE = False

EXTENSIONS = [".jpg", ".bmp", ".png", ".pgm"]

DEBUG = False


# ------------------------------------------------------------------------------
def create_samediff_traintest_split(input_path,
                                    output_prefix,
                                    # --
                                    nimgs = DEFAULT_NIMGS,
                                    ncv = DEFAULT_NCV,
                                    rseed = DEFAULT_RSEED,
                                    # --
                                    overwrite = DEFAULT_OVERWRITE,
                                    ):

    # can we even split the data ?
    assert nimgs % (2*ncv) == 0
    #assert nimgs % ncv == 0    

    seed(rseed)
    
    # -- get categories
    cat_paths = [ fil 
                  for fil in glob(input_path + "/*") 
                  if isdir(fil) ]

    cats = [ basename(cat_path) 
             for cat_path in cat_paths ]
    ncats = len(cats)

#     # can we overwrite ?
#     if exists(output_fname) and not overwrite:
#         warnings.warn("not allowed to overwrite %s"  % output_fname)
#         return

    # -- get filenames per category
    cat_fnames = {}
    for cat, cat_path in zip(cats, cat_paths):

        # get filenames with an image extension only
        cfn = [ cat + "/" + basename(fname) 
                for fname in glob(cat_path + "/*") 
                if splitext(fname)[-1].lower() in EXTENSIONS ]

#         # verify that there is enough
#         if len(cfn) < nimgs:
#             print len(cfn), ntrain+ntest
#             raise ValueError, "there is not enough images"
        
        # shuffle and get what you need
        #shuffle(cfn)
        #cfn = cfn[:ntrain+ntest]

        # split
        #cfn_train = cfn[:ntrain]
        #cfn_test = cfn[ntrain:ntrain+ntest]

        # save
        #cat_fnames[cat] = {"train": cfn_train, "test": cfn_test}
        cat_fnames[cat] = cfn

    print cat_fnames

    npos = nimgs / 2
    nneg = nimgs / 2

    # -- generate positive pairs
    pairs_pos = []
    for n in xrange(npos):
        while True:
            idx = sp.random.permutation(ncats)[0]
            cat = cats[idx]
            fns = cat_fnames[cat]
            shuffle(fns)
            pair = fns[:2]
            pair.sort()
            if pair not in pairs_pos and pair[0] != pair[1]:
                pairs_pos += [pair]
                break
        
    # -- generate negative pairs
    pairs_neg = []
    for n in xrange(npos):
        
        while True:
            
            perm = sp.random.permutation(ncats)
            
            idx1 = perm[0]
            cat1 = cats[idx1]
            fns1 = cat_fnames[cat1]
            fn1 = fns1[sp.random.permutation(len(fns1))[0]]

            idx2 = perm[1]
            cat2 = cats[idx2]
            fns2 = cat_fnames[cat2]            
            fn2 = fns2[sp.random.permutation(len(fns2))[0]]

            pair = [fn1, fn2]
            pair.sort()
            
            if pair not in pairs_neg and fn1 != fn2:
                pairs_neg += [pair]
                break


    ntest = nimgs / ncv
    ntrain = nimgs - ntest

    print "ntest=", ntest
    print "ntrain=", ntrain

    # -- generate train/test lists
    shuffle(pairs_pos)
    pairs_pos = sp.array(pairs_pos).reshape(ncv, -1, 2)
    
    shuffle(pairs_neg)
    pairs_neg = sp.array(pairs_neg).reshape(ncv, -1, 2)

    print pairs_pos.shape
    print pairs_neg.shape

    splits = []
    idx = sp.arange(ncv)
    for n in xrange(ncv):
        train_pos = pairs_pos[idx!=n].reshape(-1, 2)
        test_pos = pairs_pos[idx==n].reshape(-1, 2)
        
        train_neg = pairs_neg[idx!=n].reshape(-1, 2)
        test_neg = pairs_neg[idx==n].reshape(-1, 2)

        assert train_pos.shape[0] + train_neg.shape[0] == ntrain
        assert test_pos.shape[0] + test_neg.shape[0] == ntest

        train_l = [
            (fn[0], fn[1], '+1', 'train')
            for fn in train_pos
            ] + [
            (fn[0], fn[1], '-1', 'train')
            for fn in train_neg
            ]

        test_l = [
            (fn[0], fn[1], '+1', 'test')
            for fn in test_pos
            ] + [
            (fn[0], fn[1], '-1', 'test')
            for fn in test_neg
            ]

        split = train_l + test_l

        splits += [split]


    for n, split in enumerate(splits):
        assert len(split) == nimgs
        assert len([pair
                    for pair in split
                    if pair[3] == 'train']) == ntrain
        assert len([pair
                    for pair in split
                    if pair[3] == 'test']) == ntest
        assert len([pair
                    for pair in split
                    if pair[2] == '+1']) == npos
        assert len([pair
                    for pair in split
                    if pair[2] == '-1']) == nneg

        # -- write csv file
        out_fname = "%s%02d.csv" % (output_prefix, n+1)        
        print "="*80
        print "number of categories:", len(cats)
        print sp.array(cats)
        print "-"*80
        print "number of training examples:", ntrain
        print "number of testing examples:", ntest
        print "-"*80
        print "output filename:", out_fname
        csvw = csv.writer(open(out_fname, "w+"))
        csvw.writerows(split)

    if DEBUG:
        del csvw
        res = [e for e in csv.reader(open(output_fname)) ]
        pprint(res)
    

# ------------------------------------------------------------------------------
def main():

    usage = "usage: %prog [options] <input_path> <output_prefix>"

    parser = optparse.OptionParser(usage=usage)

    parser.add_option("--nimgs", 
                      default=DEFAULT_NIMGS,
                      type="str",
                      metavar="INT",
                      help="[default=%default]")

    parser.add_option("--ncv", 
                      default=DEFAULT_NCV,
                      type="str",
                      metavar="INT",
                      help="[default=%default]")

    parser.add_option("--rseed",
                      type="int",
                      metavar="INT",
                      default=DEFAULT_RSEED,
                      help="fix the random seed (if not None) [default=%default]")

    parser.add_option("--overwrite", 
                      default=DEFAULT_OVERWRITE,
                      action="store_true",
                      help="overwrite existing file [default=%default]")

    opts, args = parser.parse_args()

    if len(args) != 2:
        parser.print_help()
    else:
        input_path = args[0]
        output_prefix = args[1]

        create_samediff_traintest_split(input_path,
                                        output_prefix,
                                        # --
                                        nimgs = int(opts.nimgs),
                                        ncv = int(opts.ncv),
                                        rseed = opts.rseed,
                                        # --
                                        overwrite = opts.overwrite,
                                        )        


# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
