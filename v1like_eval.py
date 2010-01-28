# ------------------------------------------------------------------------------
DEBUG = False
DEFAULT_LET_ME_CHEAT = False

# ------------------------------------------------------------------------------
import os
from os import path
import csv
from pprint import pformat
import time

import numpy as np
import scipy as sp

from shogun import Kernel, Classifier, Features

# --
from pythor2.utils import funcinspect
from pythor2.utils.npprogressbar import *
widgets = [RotatingMarker(), " Progress: ", Percentage(), " ",
           Bar(left='[',right=']'), ' ', ETA()]

from pythor2.utils.mylogging import get_mylogger, LOGGING_LEVELS
logger = get_mylogger(path.split(__file__)[-1], from_pythor2=False)
# hack
DEFAULT_LOGGING_LEVEL = "debug"
logging_level = DEFAULT_LOGGING_LEVEL
logger.setLevel(LOGGING_LEVELS[logging_level.lower()])

# ------------------------------------------------------------------------------
IMAGE_EXTENSIONS = ['.png', '.jpg']
ALLOWED_EVAL_SIMFUNCS = ['abs']
LIMIT = None

DEFAULT_SVM_C = 1e5
DEFAULT_EVAL_SVM = 'LibSVM'
DEFAULT_EVAL_SIMFUNC = 'abs'

# ------------------------------------------------------------------------------
def compare(model, eval_randidx_size, fname1, fname2):
    assert fname1 != fname2
    fvector1 = _eval_process(model, fname1, eval_randidx_size)
    assert not (fvector1[0]==fvector1).all()
    fvector2 = _eval_process(model, fname2, eval_randidx_size)
    assert not (fvector2[0]==fvector2).all()
    final_fvector = similarity(fvector1, fvector2, eval_simfunc)
    return final_fvector


# ------------------------------------------------------------------------------
def _init_eval_fromdir(
    eval_dirname,              
    eval_ntrain,
    eval_ntest,
    ):

    # -- get image filenames
    eval_dirname = path.abspath(eval_dirname)
    logger.info("Image source: %s" % eval_dirname)
    
    # tree structure
    if not path.isdir(eval_dirname):
        raise ValueError, "%s is not a directory" % (eval_dirname)
    tree = os.walk(eval_dirname)
    filelist = []
    categories = tree.next()[1]    
    logger.info("Sorting files")
    for root, dirs, fnames in tree:
        assert len(fnames) == np.unique(fnames).size
        
        if dirs != []:
            msgs = ["invalid image tree structure:"]
            for d in dirs:
                msgs += ["  "+"/".join([root, d])]
            msg = "\n".join(msgs)
            raise Exception(msg)
        filelist += [ root+'/'+fname
                      for fname in fnames
                      if path.splitext(fname)[-1].lower()
                      in IMAGE_EXTENSIONS ]
    filelist.sort()
    logger.info("%d categories found:" % len(categories))
    logger.info("%s" % categories)

    # -- build eval_dict
    eval_dict = dict((cat, {}) for cat in categories)
    for cat in categories:

        cat_path = path.join(eval_dirname, cat)
        eval_dict[cat]['path'] = cat_path

        fnames = [path.split(fname)[-1]
                  for fname in filelist
                  if path.split(fname)[0] == cat_path]
                  #if fname.startswith(cat_path)]
        
        assert len(fnames) == np.unique(fnames).size
        assert len(fnames) >= eval_ntrain + eval_ntest
                
        eval_dict[cat]['filelist'] = fnames
        
    return eval_dict

# ------------------------------------------------------------------------------
eval_randidx = None
eval_cache = {}
def _eval_process(model, full_fname, eval_randidx_size):

    if DEBUG:
        return np.random.randn(100)

    # retrieve from cache?
    key = (model, full_fname, eval_randidx_size)
    if key in eval_cache:
        return eval_cache[key]
    
    arr = sp.misc.imread(full_fname).astype('float32')
    if arr.ndim == 3 and arr.shape[2] == 3:
        arr = 0.2989*arr[:,:,0] + \
              0.5870*arr[:,:,1] + \
              0.1141*arr[:,:,2]

    arr = np.atleast_3d(arr)
    arr_out = model.process(arr, resize='fit')
    fvector = arr_out[:]
    fvector = fvector.ravel()

    if eval_randidx_size is not None and fvector.size > eval_randidx_size:
        global eval_randidx
        if eval_randidx is None:
            eval_randidx = sp.random.permutation(fvector.size)
            eval_randidx = eval_randidx[:eval_randidx_size]
        fvector = fvector[eval_randidx]

    # cache it!
    eval_cache[key] = fvector.copy() # the copy is important

    assert not np.isnan(fvector).any()
    assert not np.isinf(fvector).any()

    return fvector

# ------------------------------------------------------------------------------
def v1like_eval_fromdir(
    model,
    eval_dirname,              
    eval_ntrain,
    eval_ntest,
    eval_ntrials,
    eval_rseed,
    eval_randidx_size,
    eval_svm = DEFAULT_EVAL_SVM,
    regularization = DEFAULT_SVM_C,
    ):

    logger.debug("r3ls_eval:locals() =")
    [logger.debug("%s" % line)
     for line in pformat(locals()).split('\n')]

    # -- init
    sp.random.seed(eval_rseed)
    
    eval_dict = _init_eval_fromdir(eval_dirname, eval_ntrain, eval_ntest)
    ncategories = len(eval_dict)

    full_fname0 = path.join(eval_dict[eval_dict.keys()[0]]['path'],
                            eval_dict[eval_dict.keys()[0]]['filelist'][0])
    fvector = _eval_process(model, full_fname0, eval_randidx_size)
    logger.debug_local("fvector.size")        

    train_features = np.empty((ncategories * eval_ntrain, fvector.size),
                              dtype='float32')
    test_features = np.empty((ncategories * eval_ntest, fvector.size),
                             dtype='float32')
    # -- evaluate perf
    perfs = []
    for ntrial in xrange(eval_ntrials):
        
        # - shuffle
        logger.info("Shuffling train / test sets")
        for cat_name, cat_dict in eval_dict.iteritems():

            filelist = cat_dict['filelist']

            sp.random.shuffle(filelist)
            curr_train_fl = filelist[:eval_ntrain]
            curr_test_fl = filelist[eval_ntrain:eval_ntrain+eval_ntest]
            assert len(set(curr_train_fl) & set(curr_test_fl)) == 0

            cat_dict['curr_train_fl'] = curr_train_fl
            cat_dict['curr_test_fl'] = curr_test_fl
            
        # - train
        logger.info("Collecting training data")
        start = time.time()
        train_labels = []    
        idx = 0
        pbar = ProgressBar(widgets=widgets, maxval=ncategories*eval_ntrain)
        pbar.start()        
        for cat_name, cat_dict in eval_dict.iteritems():
            curr_train_fl = cat_dict['curr_train_fl']
            cat_path = cat_dict['path']
            for fname in curr_train_fl:
                full_fname = path.join(cat_path, fname)                
                fvector = _eval_process(model, full_fname, eval_randidx_size)
                assert fvector.size == train_features.shape[1]
                train_features[idx] = fvector
                train_labels += [cat_name]
                idx += 1
                pbar.update(idx)
        train_labels = np.array(train_labels)
        pbar.finish()
        end = time.time()
        logger.info("Time: %s" % (end-start))
        logger.info("Time/image: %f" % (1.*(end-start)/train_labels.size))

        assert not np.isnan(train_features).any()
        assert not np.isinf(train_features).any()

        start = time.time()
        logger.info("Computing normalization vectors")
        fmean = train_features.mean(0)
        fstd = train_features.std(0)
        np.putmask(fstd, fstd==0, 1)
        end = time.time()
        logger.info("Time: %s" % (end-start))

        logger.info("Normalizing training data")
        start = time.time()
        train_features -= fmean
        train_features /= fstd
        end = time.time()
        logger.info("Time: %s" % (end-start))

        if eval_svm == 'LibSVM':

            logger.info("Computing traintrain linear kernel")
            start = time.time()
            kernel_traintrain = np.dot(train_features, train_features.T)
            ktrace = kernel_traintrain.trace()
            kernel_traintrain /= ktrace
            end = time.time()
            logger.info("Time: %s" % (end-start))

            alphas = {}
            support_vectors = {}
            biases = {}

            logger.info("Set traintrain custom kernel")
            start = time.time()
            customkernel = Kernel.CustomKernel()
            customkernel.set_full_kernel_matrix_from_full(kernel_traintrain.T.astype('float64'))
            end = time.time()
            logger.info("Time: %s" % (end-start))

            logger.info("Train LibSVM (C=%e)" % regularization)
            start = time.time()

            cat_index = {}

            categories = np.unique(train_labels)
            if categories.size == 2:
                categories = [categories[0]]

            for icat, cat in enumerate(categories):
                logger.debug("train SVM for '%s'" % cat)
                ltrain = np.zeros((train_labels.size))
                ltrain[train_labels != cat] = -1
                ltrain[train_labels == cat] = +1
                ltrain = ltrain.astype('float64')
                current_labels = Features.Labels(ltrain)
                svm = Classifier.LibSVM(regularization, 
                                        customkernel, 
                                        current_labels)
                assert(svm.train())
                alphas[cat] = svm.get_alphas()
                svs = svm.get_support_vectors()
                support_vectors[cat] = svs
                biases[cat] = svm.get_bias()
                cat_index[cat] = icat 

            end = time.time()
            logger.info("Time: %s" % (end-start))

        elif eval_svm == 'LibLinear':

            logger.info("Train LibLinear (C=%e)" % regularization)
            start = time.time()

            categories = np.unique(train_labels)
            if categories.size == 2:
                categories = [categories[0]]

            weights = {}
            biases = {}
            for icat, cat in enumerate(categories):
                logger.debug("train SVM for '%s'" % cat)                
                ltrain = np.zeros((train_labels.size))
                ltrain[train_labels != cat] = -1
                ltrain[train_labels == cat] = +1
                ltrain = ltrain.astype('float64')
                current_labels = Features.Labels(ltrain)

                sparse_train_features = Features.SparseRealFeatures()
                svm = Classifier.LibLinear(regularization,
                                           sparse_train_features,
                                           current_labels)
                svm.set_bias_enabled(True)

                sparse_train_features.obtain_from_simple(
                    Features.RealFeatures(train_features.T.astype("float64")))

                assert(svm.train())
                clas_w = svm.get_w()
                clas_b = np.float32(svm.get_bias())

                weights[cat] = clas_w
                biases[cat] = clas_b
                
            end = time.time()
            logger.info("Time: %s" % (end-start))
            
        else:
            raise ValueError("eval_svm '%s' not understood" % eval_svm)

        # - test
        logger.info("Collecting testing data")
        start = time.time()
        test_labels = []    
        idx = 0
        pbar = ProgressBar(widgets=widgets, maxval=ncategories*eval_ntest)
        pbar.start()
        for cat_name, cat_dict in eval_dict.iteritems():
            curr_test_fl = cat_dict['curr_test_fl']
            cat_path = cat_dict['path']
            for fname in curr_test_fl:
                full_fname = path.join(cat_path, fname)
                #arr = np.atleast_3d(sp.misc.imread(full_fname)).mean(2)
                fvector = _eval_process(model, full_fname, eval_randidx_size)
                assert fvector.size == test_features.shape[1]
                fvector -= fmean
                fvector /= fstd
                test_features[idx] = fvector
                test_labels += [cat_name]                
                idx += 1
                pbar.update(idx)
        test_labels = np.array(test_labels)
        pbar.finish()
        end = time.time()
        logger.info("Time: %s" % (end-start))        
        logger.info("Time/image: %f" % (1.*(end-start)/test_labels.size))

        assert not np.isnan(test_features).any()
        assert not np.isinf(test_features).any()

        if eval_svm == 'LibSVM':

            logger.info("Computing traintest linear kernel")
            start = time.time()
            kernel_traintest = np.dot(train_features, test_features.T)
            kernel_traintest /= ktrace
            end = time.time()
            logger.info("Time: %s" % (end-start))        


            logger.info("Testing")
            start = time.time()
            pred = np.zeros((ncategories*eval_ntest))
            distances = np.zeros((ncategories*eval_ntest, len(categories)))
            for icat, cat in enumerate(categories):        
                for point in xrange(ncategories*eval_ntest):
                    index_sv = support_vectors[cat]
                    resp = np.dot(alphas[cat], 
                                  kernel_traintest[index_sv, point]) \
                                  + biases[cat]
                    distances[point, icat] = resp

            end = time.time()
            logger.info("Time: %s" % (end-start))

        elif eval_svm == 'LibLinear':

            logger.info("Testing")
            start = time.time()
            pred = np.zeros((ncategories*eval_ntest))
            distances = np.zeros((ncategories*eval_ntest, len(categories)))
            for icat, cat in enumerate(categories):        
                for point in xrange(ncategories*eval_ntest):
                    clas_w = weights[cat]
                    clas_b = biases[cat]
                    resp = np.dot(clas_w, test_features[point]) + clas_b
                    distances[point, icat] = resp

            end = time.time()
            logger.info("Time: %s" % (end-start))

        else:
            raise ValueError("eval_svm '%s' not understood" % eval_svm)

        if len(categories) > 1:
            pred = distances.argmax(1)
            gt = np.array([cat_index[e] for e in test_labels]).astype("int")
            perf = (pred == gt)
            accuracy = 100.*perf.sum() / (ncategories*eval_ntest)
        else:
            pred = np.sign(distances).ravel()
            gt = np.array(test_labels)
            cat = categories[0]
            gt[gt != cat] = -1
            gt[gt == cat] = +1            
            gt = gt.astype("int")
            perf = (pred == gt)
            accuracy = 100.*perf.sum() / (ncategories*eval_ntest)

        logger.info("Classification accuracy on test data: %.2f " % accuracy)
        
        perfs += [accuracy]
        
                
    # --
    perfs = np.array(perfs)
    return perfs
    
# ------------------------------------------------------------------------------
def v1like_eval_fromcsv(
    model,
    eval_dirname,
    eval_csv,
    eval_rseed,
    eval_randidx_size,
    eval_svm = DEFAULT_EVAL_SVM,
    regularization = DEFAULT_SVM_C,
    eval_simfunc = DEFAULT_EVAL_SIMFUNC,
    let_me_cheat = DEFAULT_LET_ME_CHEAT, # for debugging purposes!
    ):

    # -- init
    sp.random.seed(eval_rseed)

    logger.debug("%s:locals() =" % funcinspect.whoami())
    for line in pformat(locals()).split('\n'):
        logger.debug("%s" % line)

    data_dirname = os.environ.get('V1LIKE_PT2_DATA')
    if data_dirname is None:
        raise ValueError("V1LIKE_PT2_DATA environment variable not set!")
    input_csv_fname = path.join(data_dirname, eval_csv)

    logger.info("Processing %s " % input_csv_fname)
    csvr = csv.reader(open(input_csv_fname))
    rows = [ row for row in csvr ]
    ncols = len(rows[0])

    # XXX: we need some refactoring here

    # -- nclass protocol ? (XXX: to be tested)
    if ncols == 3:
        logger.info("'nclass' evaluation protocol detected")
        ori_train_fnames = [ row[0]
                             for row in rows
                             if row[2] == "train" ][:LIMIT]
        train_fnames = [ path.join(input_path, fname+input_suffix) 
                         for fname in ori_train_fnames ][:LIMIT]
        train_labels = [ row[1]
                         for row in rows
                         if row[2] == "train" ][:LIMIT]

        ori_test_fnames = [ row[0]
                            for row in rows
                            if row[2] == "test" ][:LIMIT]
        test_fnames = [ path.join(input_path, fname+input_suffix) 
                        for fname in ori_test_fnames ][:LIMIT]
        test_labels = [ row[1]
                        for row in rows
                        if row[2] == "test" ][:LIMIT]

        raise NotImplementedError

    # -- same / not same protocol ?
    elif ncols == 4:
        logger.info("'same / different' evaluation protocol detected")
        ori_train_fnames = [ row[:2]
                             for row in rows
                             if row[3] == "train" ][:LIMIT]            
        train_pairs = [ (path.join(eval_dirname, fname1),
                         path.join(eval_dirname, fname2))
                        for fname1, fname2 in ori_train_fnames ][:LIMIT]
        train_labels = np.array([ str(row[2])
                                  for row in rows
                                  if row[3] == "train" ])[:LIMIT]            
        assert np.unique(train_labels).size > 1
        ntrain = len(train_pairs)

        ori_test_fnames = [ row[:2]
                            for row in rows
                            if row[3] == "test" ][:LIMIT]            
        test_pairs = [ (path.join(eval_dirname, fname1),
                        path.join(eval_dirname, fname2))
                       for fname1, fname2 in ori_test_fnames ][:LIMIT]
        test_labels = np.array([ str(row[2])
                                 for row in rows
                                 if row[3] == "test" ])[:LIMIT]
        assert np.unique(train_labels).size > 1
        ntest = len(test_pairs)

        # -- init data containers
        full_fname0 = train_pairs[0][0]
        fvector = _eval_process(model, full_fname0, eval_randidx_size)
        logger.debug_local("fvector.size")

        train_features = np.empty((ntrain, fvector.size), dtype='float32')
        test_features = np.empty((ntest, fvector.size), dtype='float32')

        # -- train
        logger.info("Collecting training data")
        start = time.time()
        idx = 0
        pbar = ProgressBar(widgets=widgets, maxval=ntrain)
        pbar.start()        
        for fname1, fname2 in train_pairs:
            assert fname1 != fname2
            fvector1 = _eval_process(model, fname1, eval_randidx_size)
            assert not (fvector1[0]==fvector1).all()
            fvector2 = _eval_process(model, fname2, eval_randidx_size)
            assert not (fvector2[0]==fvector2).all()
            final_fvector = similarity(fvector1, fvector2, eval_simfunc)
            train_features[idx] = final_fvector.ravel()
            if let_me_cheat:
                train_features[idx] += int(train_labels[idx])
            idx += 1
            pbar.update(idx)
        pbar.finish()

        end = time.time()
        logger.info("Time: %s" % (end-start))
        logger.info("Time/image: %f" % (1.*(end-start)/train_labels.size))

        assert not np.isnan(train_features).any()
        assert not np.isinf(train_features).any()

        start = time.time()
        logger.info("Computing normalization vectors")
        fmean = train_features.mean(0)
        fstd = train_features.std(0)
        np.putmask(fstd, fstd==0, 1)
        end = time.time()
        logger.info("Time: %s" % (end-start))

        logger.info("Normalizing training data")
        start = time.time()
        train_features -= fmean
        train_features /= fstd
        end = time.time()
        logger.info("Time: %s" % (end-start))

        assert not np.isnan(train_features).any()
        assert not np.isinf(train_features).any()

        if eval_svm == 'LibSVM':

            libsvm_train_out = _libsvm_train(train_features,
                                             train_labels,
                                             regularization,
                                             )

        elif eval_svm == 'LibLinear':
            raise NotImplementedError

            logger.info("Train LibLinear (C=%e)" % regularization)
            start = time.time()

            categories = np.unique(train_labels)
            assert categories.size > 1

            if categories.size == 2:
                categories = [categories[0]]

            weights = {}
            biases = {}
            for icat, cat in enumerate(categories):
                logger.debug("train SVM for '%s'" % cat)                
                ltrain = np.zeros((train_labels.size))
                ltrain[train_labels != cat] = -1
                ltrain[train_labels == cat] = +1
                ltrain = ltrain.astype('float64')
                current_labels = Features.Labels(ltrain)

                sparse_train_features = Features.SparseRealFeatures()
                svm = Classifier.LibLinear(regularization,
                                           sparse_train_features,
                                           current_labels)
                svm.set_bias_enabled(True)

                sparse_train_features.obtain_from_simple(
                    Features.RealFeatures(train_features.T.astype("float64")))

                assert(svm.train())
                clas_w = svm.get_w()
                clas_b = np.float32(svm.get_bias())

                weights[cat] = clas_w
                biases[cat] = clas_b

            end = time.time()
            logger.info("Time: %s" % (end-start))

        else:
            raise ValueError("eval_svm '%s' not understood" % eval_svm)

        # -- test
        logger.info("Collecting testing data")
        start = time.time()
        idx = 0
        pbar = ProgressBar(widgets=widgets, maxval=ntest)
        pbar.start()
        for fname1, fname2 in test_pairs:
            assert fname1 != fname2
            fvector1 = _eval_process(model, fname1, eval_randidx_size)
            assert not (fvector1[0]==fvector1).all()
            fvector2 = _eval_process(model, fname2, eval_randidx_size)
            assert not (fvector2[0]==fvector2).all()
            final_fvector = similarity(fvector1, fvector2, eval_simfunc)                
            test_features[idx] = final_fvector.ravel()
            if let_me_cheat:
                test_features[idx] += int(test_labels[idx])
            idx += 1
            pbar.update(idx)
        pbar.finish()
        end = time.time()
        logger.info("Time: %s" % (end-start))        
        logger.info("Time/image: %f" % (1.*(end-start)/test_labels.size))

        assert not np.isnan(test_features).any()
        assert not np.isinf(test_features).any()

        logger.info("Normalizing testing data")
        start = time.time()
        test_features -= fmean
        test_features /= fstd
        end = time.time()
        logger.info("Time: %s" % (end-start))

        assert not np.isnan(test_features).any()
        assert not np.isinf(test_features).any()

        if eval_svm == 'LibSVM':

            libsvm_test_out = _libsvm_test(train_features,
                                           test_features,
                                           libsvm_train_out['ktrace'],
                                           libsvm_train_out['categories'],
                                           libsvm_train_out['alphas'],
                                           libsvm_train_out['support_vectors'],
                                           libsvm_train_out['biases'])

        elif eval_svm == 'LibLinear':
            raise NotImplementedError

            logger.info("Testing")
            start = time.time()
            pred = np.zeros((ncategories*eval_ntest))
            distances = np.zeros((ncategories*eval_ntest, len(categories)))
            for icat, cat in enumerate(categories):        
                for point in xrange(ncategories*eval_ntest):
                    clas_w = weights[cat]
                    clas_b = biases[cat]
                    resp = np.dot(clas_w, test_features[point]) + clas_b
                    distances[point, icat] = resp

            end = time.time()
            logger.info("Time: %s" % (end-start))

        else:
            raise ValueError("eval_svm '%s' not understood" % eval_svm)

        categories = libsvm_train_out['categories']
        assert len(categories) == 1
        outputs = libsvm_test_out['outputs']

        # (assuming 2-class problem)
        # predictions
        pred = np.sign(outputs).ravel()

        # ground truth
        gt = np.empty_like(pred)            
        cat = categories[0]
        idx = test_labels == cat
        gt[idx] = +1
        gt[-idx] = -1

        # accuracy
        correct_predictions = (pred == gt)
        accuracy = 100.*correct_predictions.sum() / correct_predictions.size

    # error
    else:
        raise ValueError("csv file not understood")

    out = dict(accuracy = accuracy,
               libsvm_train_out = libsvm_train_out,
               libsvm_test_out = libsvm_test_out)

    return out

# ------------------------------------------------------------------------------
def _libsvm_train(
    train_features,
    train_labels,
    regularization,
    ):

    train_npoints = len(train_features)

    logger.info("Computing traintrain linear kernel")
    start = time.time()
    kernel_traintrain = np.dot(train_features, train_features.T)
    ktrace = kernel_traintrain.trace()
    ktrace = ktrace != 0 and ktrace or 1
    kernel_traintrain /= ktrace
    end = time.time()
    logger.info("Time: %s" % (end-start))

    alphas = {}
    support_vectors = {}
    biases = {}

    logger.info("Set traintrain custom kernel")
    start = time.time()
    customkernel = Kernel.CustomKernel()
    customkernel.set_full_kernel_matrix_from_full(kernel_traintrain.T.astype('float64'))
    end = time.time()
    logger.info("Time: %s" % (end-start))

    logger.info("Train LibSVM (C=%e)" % regularization)
    start = time.time()

    cat_index = {}

    categories = np.unique(train_labels)
    assert categories.size > 1
    if categories.size == 2:
        categories = [categories[0]]

    for icat, cat in enumerate(categories):
        logger.debug("train SVM for '%s'" % cat)
        ltrain = np.zeros((train_labels.size))
        ltrain[train_labels != cat] = -1
        ltrain[train_labels == cat] = +1
        ltrain = ltrain.astype('float64')
        current_labels = Features.Labels(ltrain)
        svm = Classifier.LibSVM(regularization, 
                                customkernel, 
                                current_labels)
        assert(svm.train())
        alphas[cat] = svm.get_alphas()
        svs = svm.get_support_vectors()
        support_vectors[cat] = svs
        biases[cat] = svm.get_bias()
        cat_index[cat] = icat
    end = time.time()
    logger.info("Time: %s" % (end-start))

    logger.info("Collecting %d training outputs" % train_npoints)
    start = time.time()
    outputs = np.zeros((train_npoints, len(categories)), dtype='float32')
    for icat, cat in enumerate(categories):        
        for point in xrange(train_npoints):
            index_sv = support_vectors[cat]
            resp = np.dot(alphas[cat], 
                          kernel_traintrain[index_sv, point]) \
                          + biases[cat]
            outputs[point, icat] = resp
    end = time.time()
    logger.info("Time: %s" % (end-start))

    out = dict(categories = categories,
               kernel_traintrain = kernel_traintrain,
               ktrace = ktrace,
               alphas = alphas,
               support_vectors = support_vectors,
               biases = biases,
               outputs = outputs)

    return out
    
#     return (categories,
#             kernel_traintrain, ktrace,
#             alphas, support_vectors, biases,
#             outputs)

# ------------------------------------------------------------------------------
def _libsvm_test(
    train_features,
    test_features,
    ktrace,
    categories,
    alphas, support_vectors, biases,
    ):

    assert ktrace != 0

    test_npoints = len(test_features)

    logger.info("Computing traintest linear kernel")
    start = time.time()

    kernel_traintest = np.dot(train_features, test_features.T)

    assert not np.isnan(kernel_traintest).any()
    assert not np.isinf(kernel_traintest).any()

    kernel_traintest /= ktrace
    end = time.time()
    logger.info("Time: %s" % (end-start))        

    assert not np.isnan(kernel_traintest).any()
    assert not np.isinf(kernel_traintest).any()
    
    logger.info("Collecting %d testing outputs" % test_npoints)
    start = time.time()
    outputs = np.zeros((test_npoints, len(categories)), dtype='float32')
    for icat, cat in enumerate(categories):        
        for point in xrange(test_npoints):
            index_sv = support_vectors[cat]
            resp = np.dot(alphas[cat], 
                          kernel_traintest[index_sv, point]) \
                          + biases[cat]
            outputs[point, icat] = resp

    end = time.time()
    logger.info("Time: %s" % (end-start))

    out = dict(kernel_traintest = kernel_traintest,
               outputs = outputs)

    return out

    
# ------------------------------------------------------------------------------
def similarity(fvector1,
               fvector2,
               eval_simfunc,
               ):

    assert eval_simfunc in ALLOWED_EVAL_SIMFUNCS
    
    if eval_simfunc == 'abs':
        out = np.absolute(fvector1-fvector2)
    else:
        raise NotImplementedError

    # XXX: Idea chi2 on a N-bin histogram

    return out
    
        

    

#     if simfunc == 'cc':
#         im1r -= im1r.mean()
#         im1r /= linalg.norm(im1r)
#         im2r -= im2r.mean()
#         im2r /= linalg.norm(im2r)
#         rep = array([dot(im1r,im2r)])
#     elif simfunc == 'sd':
#         im1r -= im1r.mean()
#         im1r /= linalg.norm(im1r)
#         im2r -= im2r.mean()
#         im2r /= linalg.norm(im2r)
#         rep = (im1r-im2r)**2.
#     elif simfunc == 'sdnonorm':
#         im1r -= im1r.mean()
#         im2r -= im2r.mean()
#         rep = (im1r-im2r)**2.
#     elif simfunc == 'sdnomean':
#         im1r -= im1r.mean()
#         im2r -= im2r.mean()
#         rep = (im1r-im2r)**2.
#     elif simfunc == 'sdnomeannonorm':
#         rep = (im1r-im2r)**2.
#     elif simfunc == 'abs':
#         im1r -= im1r.mean()
#         im1r /= linalg.norm(im1r)
#         im2r -= im2r.mean()
#         im2r /= linalg.norm(im2r)
#         rep = abs(im1r-im2r)
#     elif simfunc == 'absnonorm':
#         im1r -= im1r.mean()
#         im2r -= im2r.mean()
#         rep = abs(im1r-im2r)
#     elif simfunc == 'absnomeannonorm':
#         rep = abs(im1r-im2r)
#     elif simfunc == 'sqrtabsnomeannonorm':
#         rep = sqrt(abs(im1r-im2r))
#     elif simfunc == '28absnomeannonorm':
#         rep = ((im1r-im2r)**2.)**(1./8)
#     elif simfunc == 'logabsnomeannonorm':
#         rep = log(abs(im1r-im2r))
#     elif simfunc == 'absnomean':
#         im1r /= linalg.norm(im1r)
#         im2r /= linalg.norm(im2r)
#         rep = abs(im1r-im2r)
#     elif simfunc == 'l2':
#         rep = array([linalg.norm(im1r-im2r)])
#     elif simfunc == 'chisquare':
#         #im1r -= im1r.mean()
#         #im1r /= linalg.norm(im1r)
#         #im2r -= im2r.mean()
#         #im2r /= linalg.norm(im2r)
#         rep = array([stats.chisquare(im1r,im2r)[0]])
#     elif simfunc == 'chisquarep':
#         im1r -= im1r.mean()
#         im1r /= linalg.norm(im1r)
#         im2r -= im2r.mean()
#         im2r /= linalg.norm(im2r)
#         rep = array([stats.chisquare(im1r,im2r)])
#     else:
#         ValueError, "simfunc not understood"
