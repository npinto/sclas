
from numpy.testing import *

import numpy as np
import scipy as sp
from scipy import io, misc
import cPickle as pkl  

import sys, os
from os import path
my_path = path.dirname(path.abspath(__file__))
main_path = path.join(my_path, "..")
sys.path = [main_path] + sys.path
print path.join(my_path, "test_imageset")
imageset_path = path.join(my_path, "test_imageset")

WRITE_GROUNDTRUTH = False

def setup():
    # XXX: get dataset from the web instead
    assert path.exists(imageset_path)

    import glob
    fnames = glob.glob(path.join(imageset_path, "*/*.jpg"))

    for fname in fnames:
        dst = fname + '.pixels.mat'
        if not path.exists(dst):
            print "writing", dst
            img = misc.imread(fname)
            imgr = misc.imresize(img, (100,100))
            data = imgr.ravel().astype('float32')
            io.savemat(dst, {'data': data}, format='4')

def test_tmp():

    from tmp import kernel_generate_fromcsv_main as thefunc


    input_csv_fname = path.join(imageset_path, "train5test5_split01.csv")
    fname = "test_imageset_train5test5_split01.csv.kernel.pixels.default.mat"
    output_fname = path.join(my_path, fname)
    input_suffix = ".pixels.mat"
    input_path = imageset_path

    if path.exists(output_fname):
        os.unlink(output_fname)        

    thefunc(
        input_csv_fname,
        output_fname,
        input_suffix,
        input_path,
        )

    data = io.loadmat(output_fname)

    # -- make sure all the necessary arrays/lists are here
    thevars = {}
    keys = ['train_labels',
            'test_labels',
            'train_fnames',
            'test_fnames',
            'kernel_traintest',
            'kernel_traintrain']
    for key in keys:        
        assert key in data.keys(), "'%s' not in mat file" % key
        thevars.update(dict([(key, data[key])]))

    if WRITE_GROUNDTRUTH:
        fout = open(path.join(path.join(my_path, "test_groundtruth", fname+'.pkl'), "w+"))
        pkl.dump(thevars, fout, 2)
        fout.close()

    fin = open(path.join(path.join(my_path, "test_groundtruth", fname+'.pkl')))
    gt = pkl.load(fin)

    for key in keys:
        assert_array_equal(data[key], gt[key], "problem with '%s'" % key)

    os.unlink(output_fname)
    
if __name__ == "__main__":
    run_module_suite()
