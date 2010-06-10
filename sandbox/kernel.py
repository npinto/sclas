import scipy as sp
import numexpr as ne

# # ------------------------------------------------------------------------------
# def get_kernel_traintrain(feat1, kernel_type):
    
#     if kernel_type == "dot":
#         kernel = dot_fromfeatures(feat1, feat2)
#     elif kernel_type == "ndot":
#         kernel = ndot_fromfeatures(feat1, feat2)
#     elif kernel_type == "exp_mu_chi2":
#         chi2_matrix = chi2_fromfeatures(feat1, feat2)
#         if feat1 is not feat2:
            
#         if extra is None:
#             chi2_mu_train = chi2_matrix.mean()
#             extra = dict(chi2_mu_train=chi2_mu_train)
#         else:
#             chi2_mu_train = extra['chi2_mu_train']
#         kernel = ne.evaluate("exp(-chi2_matrix/chi2_mu_train)")
#     elif kernel_type == "exp_mu_da":
#         da_matrix = da_fromfeatures(train_features)
#         if extra is None:
#             da_mu_train = da_matrix.mean()
#             extra = dict(da_mu_train=da_mu_train)
#         else:
#             da_mu_train = extra['da_mu_train']
#         kernel = ne.evaluate("exp(-da_matrix/da_mu_train)")

#     assert(not (kernel==0).all())
#     return kernel, extra

# ------------------------------------------------------------------------------
def chi2_kernel(features1,
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
def da_kernel(features1,
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
DOT_MAX_NDIMS = 10000
def dot_kernel(features1,
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
def ndot_kernel(features1,
                features2 = None):

    features1.shape = features1.shape[0], -1
    features1 = features1/sp.sqrt((features1**2.).sum(1))[:,None]

    if features2 is None:
        features2 = features1
    else:
        features2.shape = features2.shape[0], -1
        features2 = features2/sp.sqrt((features2**2.).sum(1))[:,None]

    return sp.dot(features1, features2.T)




