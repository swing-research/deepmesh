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


idstring = sys.argv[1]
name = idstring + '.png'

P = np.load('p350.npy')
out = np.load(idstring+'_out350.npy')
mask = np.load('mask.npy').reshape(128,128)
orig = np.load('geo_originals.npy')[:,:,:,0]
print("P.shape:%s"%str(P.shape))
print("out.shape:%s"%str(out.shape))

ntest = len(out)
print(out.shape)
print('Running PLS over %d images'%ntest)
x_est = np.zeros((ntest, 128,128))

pflat = P.reshape(-1,16384)
print("Number of test samples:%d"%ntest)
# for k in range(ntest):
for k in [1]:
	c = np.zeros(50*350)

	for i in range(350):
		c[i*50:(i+1)*50] = np.matmul(P[i], out[k,i])

	x = np.zeros(16384)

	lr = 0.005

	for it in range(2000):
		g = np.matmul(pflat,x) - c
		x -= lr*np.matmul(pflat.T,g)

		x[x<=0] = 0.0
		x[x>=1] = 1.0
		if (it+1)%100==0:
			opt, _, _, _ = SNRab(mask*orig[k], mask*x.reshape(128,128))
			print('Iteration %d: %f'%(it,opt))
			imsave(name, mask*x.reshape(128,128))

	x_est[k] = x.reshape(128,128)
	print('finished %d images.'%(k+1))
	imsave('x_est%d.png'%k, x_est[k])

g = np.zeros((128*2, 128*ntest))
for i in range(ntest):
	g[:128, 128*i:128*(i+1)] = mask*orig[i]
	g[128:, 128*i:128*(i+1)] = mask*x_est[i]

imsave(name, g)
	
