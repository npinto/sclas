from os import path

import scipy as sp

from default import *
from pbar import *
from fvector import get_fvector

# ------------------------------------------------------------------------------
MEAN_MAX_NPOINTS = 2000
STD_MAX_NPOINTS = 2000

# ------------------------------------------------------------------------------
class OverwriteError(Exception): pass

# ------------------------------------------------------------------------------
def get_features(fnames,
                 featfunc,
                 kernel_type = DEFAULT_SIMFUNC,
                 simfunc = DEFAULT_SIMFUNC,
                 info_str = 'the',
                 ):
    
    # --------------------------------------------------------------------------
    # -- init
    # load first vector to get dimensionality
    fvector0 =  get_fvector(fnames[0],
                            featfunc,
                            kernel_type,
                            simfunc=simfunc)
    
    if kernel_type == "exp_mu_da":
        # hack for GB with 204 dims
        fvector0 = fvector0.reshape(-1, 204)
    else:
        fvector0 = fvector0.ravel()
    featshape = fvector0.shape
    featsize = fvector0.size

    # -- helper function
    # set up progress bar
    def load_features(x_fnames, info_str):        
        print "-"*80
        print "Loading %s data ..." % info_str
        pbar = ProgressBar(widgets=widgets, maxval=len(x_fnames))
        pbar.start()

        x_features = sp.empty((len(x_fnames),) + featshape,
                              dtype='float32')
        
        for i, one_or_two_fnames in enumerate(x_fnames):
            fvector = get_fvector(one_or_two_fnames,
                                  featfunc,                                  
                                  kernel_type,
                                  simfunc=simfunc)
            fvector = fvector.reshape(fvector0.shape)
            x_features[i] = fvector
            pbar.update(i+1)

        pbar.finish()
        print "-"*80        

        return x_features

    # -- load features from filenames
    try:
        features = load_features(fnames, info_str=info_str)
    except OverwriteError, err:
        print err

    assert(not sp.isnan(sp.ravel(features)).any())
    assert(not sp.isinf(sp.ravel(features)).any())
    
    return features


# ------------------------------------------------------------------------------
def sphere_features(features, sphere_vectors):
    
    features.shape = features.shape[0], -1

    fmean, fstd = sphere_vectors
    features -= fmean        
    assert((fstd!=0).all())
    features /= fstd

    assert(not sp.isnan(sp.ravel(features)).any())
    assert(not sp.isinf(sp.ravel(features)).any())
    
    return features


def get_sphere_vectors(features):

    fshape = features.shape
    features.shape = fshape[0], -1
    npoints, ndims = features.shape

    if npoints < MEAN_MAX_NPOINTS:
        fmean = features.mean(0)
    else:
        # - try to optimize memory usage...
        sel = features[:MEAN_MAX_NPOINTS]
        fmean = sp.empty_like(sel[0,:])

        sp.add.reduce(sel, axis=0, dtype="float32", out=fmean)

        curr = sp.empty_like(fmean)
        npoints_done = MEAN_MAX_NPOINTS
        while npoints_done < npoints:

            # check if can we overwrite (other process)
            if path.exists(output_fname) and not overwrite:
                warnings.warn("not allowed to overwrite %s"  % output_fname)
                return

            sel = features[npoints_done:npoints_done+MEAN_MAX_NPOINTS]
            sp.add.reduce(sel, axis=0, dtype="float32", out=curr)
            sp.add(fmean, curr, fmean)
            npoints_done += MEAN_MAX_NPOINTS                

        #fmean = features[:MEAN_MAX_NPOINTS].sum(0)
        #npoints_done = MEAN_MAX_NPOINTS
        #while npoints_done < npoints:
        #    fmean += features[npoints_done:npoints_done+MEAN_MAX_NPOINTS].sum(0)
        #    npoints_done += MEAN_MAX_NPOINTS

        fmean /= npoints

    if npoints < STD_MAX_NPOINTS:
        fstd = features.std(0)
    else:
        # - try to optimize memory usage...

        sel = features[:MEAN_MAX_NPOINTS]

        mem = sp.empty_like(sel)
        curr = sp.empty_like(mem[0,:])

        seln = sel.shape[0]
        sp.subtract(sel, fmean, mem[:seln])
        sp.multiply(mem[:seln], mem[:seln], mem[:seln])
        fstd = sp.add.reduce(mem[:seln], axis=0, dtype="float32")

        npoints_done = MEAN_MAX_NPOINTS
        while npoints_done < npoints:

            # check if can we overwrite (other process)
            if path.exists(output_fname) and not overwrite:
                warnings.warn("not allowed to overwrite %s"  % output_fname)
                return

            sel = features[npoints_done:npoints_done+MEAN_MAX_NPOINTS]
            seln = sel.shape[0]
            sp.subtract(sel, fmean, mem[:seln])
            sp.multiply(mem[:seln], mem[:seln], mem[:seln])
            sp.add.reduce(mem[:seln], axis=0, dtype="float32", out=curr)
            sp.add(fstd, curr, fstd)

            npoints_done += MEAN_MAX_NPOINTS

        # slow version:
        #fstd = ((features[:MEAN_MAX_NPOINTS]-fmean)**2.).sum(0)
        #npoints_done = MEAN_MAX_NPOINTS
        #while npoints_done < npoints:
        #    fstd += ((features[npoints_done:npoints_done+MEAN_MAX_NPOINTS]-fmean)**2.).sum(0)
        #    npoints_done += MEAN_MAX_NPOINTS

        fstd = sp.sqrt(fstd/npoints)

    fstd[fstd==0] = 1
    sphere_vectors = (fmean, fstd)
    features.shape = fshape

    return sphere_vectors

