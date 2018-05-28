"""Utility to reconstruct from subnet

Running subnet evaluation generates a coefficients numpy
array '*_out350.npy' where * is the evaluation experiment name.

This utility was hardcoded for 350 meshes. Needs some work to 
scale to variable number of meshes.

Algorithm:
We basically run an iterative non-negative least-squares method
to figure out the reconstructions. The SNR of reconstruction is 
printed for reference.
"""

import sys
import numpy as np
from SNRab import *
from scipy.misc import imsave, imread

head_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(head_dir+'/utils/')
from parser_sub_direct import parserMethod

args = parserMethod()
nproj = args.num_projectors_to_use
mask_loc = args.mask
idstring = args.recon_coefficients
orig_loc = args.recon_originals
ntri = args.dim_rand_subspace
lr = args.learning_rate
im_s = args.img_size

name = idstring + '.png'

P = np.load('p%d.npy'%nproj)
out = np.load(idstring+'_out%d.npy'%nproj)
mask = np.load(mask_loc).reshape(im_s, im_s)
orig = np.load(orig_loc)[:, :, :, 0]
print("P.shape:%s" % str(P.shape))
print("out.shape:%s" % str(out.shape))

ntest = len(out)
print(out.shape)
print('Running PLS over %d images' % ntest)
x_est = np.zeros((ntest, im_s, im_s))

pflat = P.reshape(-1, im_s*im_s)
print("Number of test samples:%d" % ntest)
for k in range(ntest):
    c = np.zeros(ntri*nproj)

    for i in range(nproj):
        c[i*ntri:(i+1)*ntri] = np.matmul(P[i], out[k, i])

    x = np.zeros(im_s*im_s)

    for it in range(2000):
        g = np.matmul(pflat, x) - c
        x -= lr*np.matmul(pflat.T, g)

        x[x <= 0] = 0.0
        x[x >= 1] = 1.0
        if (it+1) % 100 == 0:
            opt, _, _, _ = SNRab(mask*orig[k], mask*x.reshape(im_s, im_s))
            print('Iteration %d: %f' % (it, opt))
            imsave(name, mask*x.reshape(im_s, im_s))

    x_est[k] = x.reshape(im_s, im_s)
    print('finished %d images.' % (k+1))
    imsave('x_est%d.png' % k, x_est[k])

g = np.zeros((im_s*2, im_s*ntest))
for i in range(ntest):
    g[:im_s, im_s*i:im_s*(i+1)] = mask*orig[i]
    g[im_s:, im_s*i:im_s*(i+1)] = mask*x_est[i]

imsave(name, g)
