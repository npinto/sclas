#!/usr/bin/python
# -*- coding: utf-8 -*-

""" TODO: youssef docstring """

from scipy import array, io, double, zeros, dot
from shogun import Kernel, Classifier, Features

import optparse
import sys
import copy

#from numpy import unique
from scipy import unique, mgrid, trace, sign, ascontiguousarray, empty, diag
from scipy import *
import time

C_DEFAULT = 1e4
LIMIT = int(4e6)
#MKL = True

from IPython.Shell import IPShellEmbed
ipshell = IPShellEmbed(argv=[])


cache_train = {}
cache_test = {}

def mkl_train(fname_l, reg, ltrain, no_mkl, flip):

    global cache

    d0 = io.loadmat(fname_l[0])

    if flip:
        keytrain = "kernel_traintest"
        keytest = "kernel_traintrain"        
    else:
        keytrain = "kernel_traintrain"
        keytest = "kernel_traintest"

    ltr = ltrain.astype(double)
    #l = ((l==thecat)-1*(l!=thecat)).astype(double)
    labels = Features.Labels(ltr)
    
    if True:
        tr1_l = []
        tr2_l = []
        ker_l = []
        #feat_l = []
    
        combker = Kernel.CombinedKernel()
        #combfeat = Features.CombinedFeatures()
        
        m,n = d0[keytrain].shape
        yy,xx = mgrid[:m,:n]

        print "loading nfiles=", len(fname_l)
        for i,fname in enumerate(fname_l):
            if fname in cache_train:
                kermat, tr1, tr2 = cache_train[fname]
            else:
                #print "loading", fname
                d = io.loadmat(fname)
                kermat = (d[keytrain]).astype(double)
                #kermat += kermat.min()
                del d
                #kdiag = 
                #kermat /= kdiag
                tr1 = trace(kermat)*len(fname_l)#*diag(kermat)
                if tr1 == 0:
                    print "trace is 0, skipping"
                    continue
                print i+1, "trace", tr1, "min", kermat.min(), "max", kermat.max(), fname                
                kermat /= tr1
                #print kermat
                #kermat = exp(kermat*len(kermat))
                #tr2 = trace(kermat)
                #kermat /= tr2
                #print kermat
                tr2 = 0
                cache_train[fname] = kermat, tr1, tr2
                #print diag(kermat)

            ker = Kernel.CustomKernel(kermat)
            #ker.set_full_kernel_matrix_from_full(kermat)
            #feat = Features.DummyFeatures(n)
            combker.append_kernel(ker)
            #combfeat.append_feature_obj(feat)
        
            ker_l += [ker]
            #feat_l += [feat]
            tr1_l += [tr1]
            tr2_l += [tr2]

        #combker.init(combfeat, combfeat)
        cache = {'ker_l': ker_l,
                 #'feat_l': feat_l,
                 'tr1_l':tr1_l,
                 'tr2_l':tr2_l,
                 'combker': combker,
                 #'combfeat': combfeat,
                 }

        del xx, yy    

    if no_mkl:
        #w_l = array([.2, .8])
        #w_l /= w_l.sum()
        for i,ker in enumerate(ker_l):
            ker.set_combined_kernel_weight(1./(len(ker_l)))
            #ker.set_combined_kernel_weight(w_l[i])

    tstart = time.time()
    #svm1 = Classifier.SVMLight(reg, combker, labels)
    #svm1 = Classifier.LibSVM(reg, combker, labels)
    #svm1 = Classifier.GMNPSVM(reg, combker, labels)
    #svm1 = Classifier.SVMSGD(reg, combker, labels)
    #svm1 = Classifier.CPLEXSVM(reg, combker, labels)
    if len(fname_l) == 1:
        no_mkl = True
        #svm1 = Classifier.LibSVM(reg, combker, labels)
        svm1 = Classifier.SVMLight(reg, combker, labels)
    else:
        #svm1 = Classifier.SVMLight(reg, combker, labels)
        svm1 = Classifier.MKLClassification()
        svm1.set_C(reg, reg)
        svm1.set_kernel(combker)
        svm1.set_labels(labels)
        print combker.get_subkernel_weights()
        #raise
    #svm1.set_mkl_enabled(not no_mkl)
    
    #svm1.set_C(reg/10., reg)
    #print svm1.get_C1(), svm1.get_C2()

    svm1.parallel.set_num_threads(4)
    #print ker_l[0]


    kkk = combker.get_kernel_matrix()
    assert(not isnan(kkk).any())
    assert(not isinf(kkk).any())# trace(kkk)

    #svm1.set_linadd_enabled(True)
    #svm1.set_shrinking_enabled(True)
    #svm1.set_bias_enabled(False)
    #svm1.set_epsilon(1e-1)
    #svm1.set_tube_epsilon(1e-4)
    #svm1.set_weight_epsilon(0)
    #svm1.set_C_mkl(1e3)
    #ipshell()
    svm1.set_mkl_norm(1)

    assert(svm1.train())
    www = combker.get_subkernel_weights()
    print www
    print "-w " + " -w ".join([str(e) for e in www])
    sys.exit(0)
    #raise
#print 

    alphas = svm1.get_alphas()
    bias = svm1.get_bias()
    svs = svm1.get_support_vectors()
    
    #print ker_l[0]

    #svm1 = Classifier.SVMLight(reg, ker, labels)
    #svm1.train()
    #print svm1.get_alphas()

    weight_l = []
    for i,ker in enumerate(ker_l):
        weight = ker.get_combined_kernel_weight()
        weight_l += [weight]

    print weight_l
    print "-w " + " -w ".join([str(e) for e in weight_l])
    kweights = array(weight_l)
    kweights /= kweights.sum()

    traces1 = array(tr1_l)
    #traces2 = array(tr2_l)

    #raise
    
    #tend = time.time()
    #print "time:", tend-tstart
    
    #kernel_train = combker.get_kernel_matrix()
    
#     weight_l = []
#     for i,ker in enumerate(ker_l):
#         weight = ker.get_combined_kernel_weight()
#         weight_l += [weight]

    #combker.init(combfeat,combfeat)
    #print svm1.classify().get_labels()[:10]

    #kk = combker.get_kernel_matrix()
    #print "trace=", trace(kk)
    #trainperf = (sign(svm1.classify().get_labels())==ltr).mean() 
    #print "trainperf=", trainperf, weight_l

    #print svm1.get_alphas()
    #raise

#     mkl = {"svm": svm1, 
#            "ker_l": ker_l,
#            "feat_l": feat_l,
#            "tr_l": tr_l,
#            "combker": combker,
#            "combfeat": combfeat,
#            }

    #print bias

    mkl = {"alphas": alphas.astype('float32'),
           "bias": array(bias).astype('float32'),
           "svs": svs.astype('int32'),
           "kweights": kweights.astype('float32'),
           "traces1": traces1.astype('float32'),
           #"traces2": traces2.astype('float32'),
           }
    
    #print svm1.get_alphas()

    return mkl


def mkl_test(mkl, fname_l, flip):


    if flip:
        keytrain = "kernel_traintest"
        keytest = "kernel_traintrain"        
    else:
        keytrain = "kernel_traintrain"
        keytest = "kernel_traintest"

    #keytest = "kernel_train"
    d0 = io.loadmat(fname_l[0])

    #combfeattst = Features.CombinedFeatures()

    alphas = mkl["alphas"]

    #print (alphas<0).sum(), (alphas>0).sum()

    bias = mkl["bias"]
    svs = mkl["svs"]
    kweights = mkl["kweights"]
    traces1 = mkl["traces1"]
    #traces2 = mkl["traces2"]

    m,n = d0[keytrain].shape

    nkernels = len(fname_l)
    k3d = empty((m,n,nkernels), 'float32')

    for i,fname in enumerate(fname_l):
        d = io.loadmat(fname)
        kermattst = d[keytest]/traces1[i]
        #kermattst = exp(kermattst*len(kermattst))
        k3d[:,:,i] = kermattst#/traces2[i]

    kernel_test = (kweights*k3d).sum(2)
    
    
    preds = dot(alphas, kernel_test[svs, :]) + bias
    return preds
    

#-----------------------------------------------------------
def mkl_one_vs_all(fname_l,
                   reg,
                   no_mkl=False,
                   flip=False,
                   mean_mkl=False):
    
    """ TODO: youssef docstring """


    d0 = io.loadmat(fname_l[0])
    
    print d0.keys()

    labels_train = array(d0['train_labels'])
    labels_test = array(d0['test_labels'])


    categories = unique(labels_train)[:LIMIT]
    n_categories = categories.size

    perf_l = []

    mkls = []

    #gt = array([int(e) for e in labels_test])
    gt = labels_test

    for icat, cat in enumerate(categories):
        sys.stdout.write("%s."%cat)
        sys.stdout.flush()
        ltrain = array(labels_train)
        ltrain[labels_train == cat] = +1
        ltrain[labels_train != cat] = -1
        #print ltrain
        #print labels_train
        mkl = mkl_train(fname_l, reg, ltrain, no_mkl, flip)
        #mkls[icat] = mkl
        mkls += [mkl]
        #print (sign(mkl_test(mkl, fname_l))==ltrain).mean()
        #print mkl
        #print mkl['svm'].get_bias()
#         if icat == 12:
#             #print mkl['svm'].get_bias()
#             #print ltrain
#             print mkl['svm'].get_alphas()
#             #raise

    print 

    #if mean_mkl:
        

    n_test = len(labels_test)
    pred = zeros((n_test))
    distance = zeros((n_test, n_categories))
    #for point in xrange(n_test):
    for icat, cat in enumerate(categories):
        sys.stdout.write("%s."%cat)
        sys.stdout.flush()
        # pred
        mkl = mkls[icat]
        #print mkl['svm'].get_alphas()
        dd = mkl_test(mkl, fname_l, flip)
        #print dd.mean()
        #raise
        distance[:, icat] = dd
        #if icat == 12:
        #print dd[:10], dd.argmax(), (dd>0).sum(), (dd<0).sum()
        #print (dd>0).sum(), (dd<0).sum()


#     iperf = []
#     for icat, cat in enumerate(categories):
#         sys.stdout.write("%s."%int(cat))
#         sys.stdout.flush()
#         ltest = array(labels_test)
#         ltest[labels_train == cat] = +1
#         ltest[labels_train != cat] = -1
#         sd = sign(distance[:, icat])
#         ip = (sd==ltest).mean()
#         iperf += [ip]

#     print
#     print "mean(iperf)", mean(iperf)
        

    pred = [categories[i] for i in distance.argmax(1)]
    #print pred
    #print gt
    #from pylab import *
    #print distance
    #matshow(distance)
    #show()
    #print pred
    #print gt
    
    #print gt
    
#     import scipy as sp

#     print distance

#     c = distance[:,0]
#     print c
#     si = sp.argsort(-c)
#     print si
#     print sp.single(labels_test[si]>0)
#     tp = sp.cumsum(sp.single(labels_test[si]>0))
#     fp = sp.cumsum(sp.single(labels_test[si]<0))
#     rec  = tp/sp.sum(labels_test>0)
#     prec = tp/(fp+tp)
    
#     print rec
#     print prec

#     ap = 0
#     rng = sp.arange(0,1.1,.1)
#     for th in rng:
#         p = prec[rec>=th].max()
#         if p == []:
#             p = 0
#         ap += p / rng.size

#     print "Average Precision:", ap

    perf = (pred == gt).astype(double)
    #print perf
        
    perf_l += [sum(perf) / n_test]                
    perf_a = array(perf_l)
    return perf_l[0]
    
#----------------------------------------------------------
def main():

    """ TODO: youssef docstring """    

    usage = "usage: %prog [options] <kernel_mat_filename1> ... <kernel_mat_filenameN>"
    
    parser = optparse.OptionParser(usage=usage)
    
    parser.add_option("--regularization_parameter", "-C",
                      type="float",
                      default = C_DEFAULT,
                      dest="C",
                      help="[default=%default]")
  
    parser.add_option("--no_mkl",
                      #type="bool",
                      default = False,
                      action="store_true",
                      help="[default=%default]")

    parser.add_option("--mean_mkl",
                      default = False,
                      action="store_true",
                      help="[default=%default]")
  
    options, arguments = parser.parse_args()

    if len(arguments) < 1:
        parser.print_help()
    else:
        fname_l = arguments
        regularization_parameter = options.C
        
        res1 = mkl_one_vs_all(fname_l,
                              regularization_parameter,
                              no_mkl=options.no_mkl,
                              flip=False,
                              mean_mkl = options.mean_mkl)
#         res2 = mkl_one_vs_all(fname_l,
#                               regularization_parameter,
#                               no_mkl=options.no_mkl,
#                               flip=True)
        #print (res1+res2)/2.
        print
        print "perf=", res1

#-------------------------------------------------------
if __name__ == "__main__":
    main()

