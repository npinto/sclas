#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys

import scipy as sp
from scipy import io
from shogun import Kernel, Classifier, Features

regularization = 10

#svm_out_l = ["tmp%d.mat.1000.svm_out.mat" % i for i in xrange(2,10)]
svm_out_l = sys.argv[1:]

train_features = []
train_labels = None
test_features = []
test_labels = None

max_train_accuracy = 0
max_test_accuracy = 0

for svm_out in svm_out_l:

    print svm_out

    data = io.loadmat(svm_out)

    train_feat = data['train_predictions'].ravel()
    train_lab = data['train_y'].ravel()

    test_feat = data['test_predictions'].ravel()
    test_lab = data['test_y'][:,1].ravel()

    train_features += [train_feat]
    test_features += [test_feat]

    if train_labels is None:
        train_labels = train_lab
    else:
        assert((train_labels == train_lab).all())
    
    if test_labels is None:
        test_labels = test_lab
    else:
        assert((test_labels == test_lab).all())

    train_accuracy = 100.*(sp.sign(train_feat) == train_lab).sum()/train_lab.size
    print sp.sign(test_feat).size
    print test_lab.size
    test_accuracy = 100.*(sp.sign(test_feat) == test_lab).sum()/test_lab.size
    print "train accuracy:", train_accuracy
    print "test accuracy:", test_accuracy

    if max_train_accuracy < train_accuracy:
        max_train_accuracy = train_accuracy
    
    if max_test_accuracy < test_accuracy:
        max_test_accuracy = test_accuracy

print "max_train_accuracy:", max_train_accuracy
print "max_test_accuracy:", max_test_accuracy
    

train_features = sp.array(train_features)
test_features = sp.array(test_features)
ntrain = train_features.shape[0]
ntest = test_features.shape[0]

print "normalization"
fmean = train_features.mean(1)
train_features -= fmean[:, None]
test_features -= fmean[:, None]
fstd = train_features.std(1)
train_features /= fstd[:, None]
test_features /= fstd[:, None]

from mclp import LPBoostMulticlassClassifier

lp = LPBoostMulticlassClassifier(2, 0.1)
#lp.initialize_boosting([0,1], False, "clp")

for tf in train_features:
    lp.add_multiclass_classifier([tf-1,tf])

lp.update()
print "rho:", lp.rho


train_predictions = sp.zeros(train_labels.shape)
for i in xrange(len(train_features)):
    #print i, lp.weights[0][i]
    train_predictions += lp.weights[0][i]*train_features[i]
print "train perf=", 100.*(sp.sign(train_predictions)==train_labels).sum()/train_labels.size

test_predictions = sp.zeros(test_labels.shape)
for i in xrange(len(test_features)):
    #print i, lp.weights[0][i]
    test_predictions += lp.weights[0][i]*test_features[i]
print "test perf=", 100.*(sp.sign(test_predictions)==test_labels).sum()/test_labels.size


print "-w " + " -w ".join([str(lp.weights[0][i]) for i in xrange(len(test_features))])

# assert(c.weights[0][0] - 1.0 <  1e-8)
# assert(c.rho - 1.0 < 1e-8)


# kernel_traintrain = sp.dot(train_features.T, train_features)
# kernel_traintest = sp.dot(train_features.T, test_features)

# customkernel = Kernel.CustomKernel()
# customkernel.set_full_kernel_matrix_from_full(kernel_traintrain)

# print "create svm"
# svm = Classifier.LibSVM(regularization,
#                         customkernel, 
#                         Features.Labels(train_labels))

# print "train svm"
# assert(svm.train())

# alphas = svm.get_alphas()
# svs = svm.get_support_vectors()
# bias = svm.get_bias()
# print " %d SVs" % len(svs)

# # -- test
# print "Predicting training data ..."
# train_predictions = sp.dot(alphas, 
#                            kernel_traintrain[svs]) + bias
# print "perf=", 100.*(sp.sign(train_predictions)==train_labels).sum()/train_labels.size

# print "Predicting testing data ..."
# test_predictions = sp.dot(alphas, 
#                           kernel_traintest[svs]) + bias
# print "perf=", 100.*(sp.sign(test_predictions)==test_labels).sum()/test_labels.size

