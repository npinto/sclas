
from default import (
    DEFAULT_KERNEL_TYPE,
    DEFAULT_SIMFUNC,
    DEFAULT_USE_CACHE)

VALID_SIMFUNCS = [
    # -- CVPR09
    'abs_diff',
    'sq_diff',    
    'sqrtabs_diff',
    # -- NP Jan 2010
    'mul',
    'sqrt_mul',
    'sq_add',
    'pseudo_AND_soft_range01',
    'concat',
    # -- Others
    #'sq_diff_o_sum',
    # -- DDC Feb 2010
    'normalized_AND_soft', 
    #'normalized_AND_hard_0.5', # poor performance
    #'pseudo_AND_soft', # poor performance
    'pseudo_AND_hard_0.5',
    'pseudo_AND_hard_0.25',
    # -- tmp    
    'tmp',
    'tmp2',
    'tmp4',
    'tmp5',
    'tmp6',
    'tmp7',
    'tmp8',
    'tmp10',
    ]

fvector_cache = {}

# ------------------------------------------------------------------------------
def get_fvector(one_or_two_fnames,
                featfunc,
                kernel_type = DEFAULT_KERNEL_TYPE,
                simfunc = DEFAULT_SIMFUNC,
                use_cache = DEFAULT_USE_CACHE,
                ):

    global fvector_cache
    
    if len(one_or_two_fnames) == 1:
        fname = one_or_two_fnames[0]

        if fname not in fvector_cache:
            fvector = featfunc(fname)
        else:
            fvector = fvector_cache[fname].copy()

        if use_cache:
            fvector_cache[fname] = fvector.copy()
            
    elif len(one_or_two_fnames) == 2:
        fname1, fname2 = one_or_two_fnames
        if (fname1, fname2) not in fvector_cache:
            fdata1 = featfunc(fname1)
            fdata2 = featfunc(fname2)
            assert fdata1.shape == fdata2.shape, "with %s and %s" % (fname1, fname2)
            fvector = kernel_generate_fromcsv.get_simfunc_fvector(
                fdata1, fdata2, simfunc=simfunc)
        else:
            fvector = fvector_cache[(fname1, fname2)].copy()

        if use_cache:
            fvector_cache[(fname1, fname2)] = fvector.copy()            

    else:
        raise ValueError("len(one_or_two_fnames) = %d" % len(one_or_two_fnames))

    return fvector

# ------------------------------------------------------------------------------
def get_simfunc_fvector(fdata1, fdata2, simfunc=DEFAULT_SIMFUNC):

    assert simfunc in VALID_SIMFUNCS

    if simfunc == 'abs_diff':
        fvector = sp.absolute(fdata1-fdata2)

    elif simfunc == 'sq_diff':
        fvector = (fdata1-fdata2)**2.

    elif simfunc == 'sq_diff_o_sum':
        denom = (fdata1+fdata2)
        denom[denom==0] = 1
        fvector = ((fdata1-fdata2)**2.) / denom

    elif simfunc == 'sqrtabs_diff':
        fvector = sp.sqrt(sp.absolute(fdata1-fdata2))

    elif simfunc == 'mul':
        fvector = fdata1*fdata2

    elif simfunc == 'sqrt_mul':
        fvector = sp.sqrt(fdata1*fdata2)

    elif simfunc == 'sq_add':
        fvector = (fdata1 + fdata2)**2.

    elif simfunc == 'pseudo_AND_soft_range01':
        assert fdata1.min() != fdata1.max()
        fdata1 -= fdata1.min()
        fdata1 /= fdata1.max()
        assert fdata2.min() != fdata2.max()
        fdata2 -= fdata2.min()
        fdata2 /= fdata2.max()
        denom = fdata1 + fdata2
        fvector = 4. * (fdata1 / denom) * (fdata2 / denom)
        sp.putmask(fvector, sp.isnan(fvector), 0)
        sp.putmask(fvector, sp.isinf(fvector), 0)                        

    elif simfunc == 'concat':
        return sp.concatenate((fdata1, fdata2))

    # DDC additions, FWTW:
    elif simfunc == 'normalized_AND_soft':
        fvector = (fdata1 / fdata1.std()) * (fdata2 / fdata2.std())

    elif simfunc == 'normalized_AND_hard_0.5':
        fvector = ((fdata1 / fdata1.std()) * (fdata2 / fdata2.std()) > 0.5)

    elif simfunc == 'pseudo_AND_soft':
        # this is very similar to mul.  I think it may be one "explanation" for why mul is good
        denom = fdata1 + fdata2
        # goes from 1 when fdata1==fdata2, to 0 when they are very different
        fvector = 4. * (fdata1 / denom) * (fdata2 / denom)  
        fvector[sp.isnan(fvector)] = 1 # correct behavior is to have the *result* be one
        fvector[sp.isinf(fvector)] = 1

    elif simfunc == 'pseudo_AND_hard_0.5':
        denom = fdata1 + fdata2
        # goes from 1 when fdata1==fdata2, to 0 when they are very different
        fvector = ( (4. * (fdata1 / denom) * (fdata2 / denom)) > 0.5 )        

    elif simfunc == 'pseudo_AND_hard_0.25':
        denom = fdata1 + fdata2
        # goes from 1 when fdata1==fdata2, to 0 when they are very different
        fvector = ( (4. * (fdata1 / denom) * (fdata2 / denom)) > 0.25 )

    elif simfunc == 'tmp':
        fvector = fdata1**2. + fdata2**2.

    elif simfunc == 'tmp2':
        fvector = fdata1**2. + fdata1*fdata2 + fdata2**2.

    #elif simfunc == 'pseudo_AND_soft':
    elif simfunc == 'tmp4':
        # this is very similar to mul.  I think it may be one "explanation" for why mul is good
        denom = fdata1 + fdata2
        denom[denom==0] = 1
        # goes from 1 when fdata1==fdata2, to 0 when they are very different
        fvector = 4. * (fdata1 / denom) * (fdata2 / denom)          

    #elif simfunc == 'pseudo_AND_hard_0.5':
    elif simfunc == 'tmp5':        
        denom = fdata1 + fdata2
        denom[denom==0] = 1
        # goes from 1 when fdata1==fdata2, to 0 when they are very different
        fvector = ( (4. * (fdata1 / denom) * (fdata2 / denom)) > 0.5 )

    #elif simfunc == 'pseudo_AND_hard_0.25':
    elif simfunc == 'tmp6':                
        denom = fdata1 + fdata2
        denom[denom==0] = 1
        # goes from 1 when fdata1==fdata2, to 0 when they are very different
        fvector = ( (4. * (fdata1 / denom) * (fdata2 / denom)) > 0.25 )

    elif simfunc == 'tmp7':                
        denom = fdata1 + fdata2
        denom[denom==0] = 1
        # goes from 1 when fdata1==fdata2, to 0 when they are very different
        fvector = ( (4. * (fdata1 / denom) * (fdata2 / denom)) > 0.1 )            
    elif simfunc == 'tmp8':
        #assert fdata1.min() != fdata1.max()
        #fdata1 -= fdata1.min()
        #fdata1 /= fdata1.max()
        #assert fdata2.min() != fdata2.max()
        #fdata2 -= fdata2.min()
        #fdata2 /= fdata2.max()
        denom = fdata1 + fdata2
        fvector = 4. * (fdata1 / denom) * (fdata2 / denom)
        #sp.putmask(fvector, sp.isnan(fvector), 0)
        fvector[sp.isnan(fvector)] = 0
        fvector[sp.isinf(fvector)] = 0
        assert(not sp.isnan(fvector).any())

    elif simfunc == 'tmp10':
        assert fdata1.min() != fdata1.max()
        fdata1 -= fdata1.min()
        fdata1 /= fdata1.max()
        assert fdata2.min() != fdata2.max()
        fdata2 -= fdata2.min()
        fdata2 /= fdata2.max()
        denom = fdata1 + fdata2
        #fvector = 4. * (fdata1 / denom) * (fdata2 / denom)
        fvector = ( (4. * (fdata1 / denom) * (fdata2 / denom)) > 0.25 )
        #sp.putmask(fvector, sp.isnan(fvector), 0)
        fvector[sp.isnan(fvector)] = 0
        fvector[sp.isinf(fvector)] = 0
        assert(not sp.isnan(fvector).any())

    return fvector
        
