
from __future__ import division
from __future__ import print_function
from __future__ import division
from __future__ import print_function

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"



import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# %tensorflow_version 1.14
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy.linalg as la
import scipy.io as sio
import math
import sys
import time
import pdb
import matplotlib.pyplot as plt
import problem, network, train
import blocksparsetoolbox as bst

from scipy.linalg import toeplitz, circulant

np.random.seed(1)
tf.set_random_seed(1)

def R(D, n, d):
    R = np.zeros((n,n))
    for k in range(0,n):
        for l in range(0,n):
            if k != l:
                R[k,l] = np.linalg.norm(D[:, k*d:(k+1)*d].T@D[:, l*d:(l+1)*d], 2)
    return R
    
case = 10 
n = 128
m = 32
d = 15
K = np.zeros((m,n))

for l in range(0,n):
        K[:,l:(l+1)] = np.random.normal(size=(m, 1))
        K[:,l:(l+1)], _ = np.linalg.qr(K[:,l:(l+1)])
        
        
print('rank(D^T@D)='+str(np.linalg.matrix_rank(K.T@K)))
print('rank(D@D^T)='+str(np.linalg.matrix_rank(K@K.T)))
print('rank(D)='+str(np.linalg.matrix_rank(K)))
print('coherence = '+str(np.max(np.abs(K.T@K-np.eye(n)))))
print(la.norm(K,2))

SNR = np.infty # signal to noise ratio given in dB
MC = 250 # batch number
prob = problem.mmv_problem(K,B=d, MC=MC, pnz=0.1, SNR_dB=SNR) 

# creating the network and setup training:

T = 6 # number of layers/iterations

# computing the analytical weight matrix for ALBISTA

W = bst.compute_W_Thm(prob.A_s, prob.L, 1)
print('cross-coherence = '+str(np.max(np.abs(W.T@prob.A_s-np.eye(n)))))
layers = network.build_ALBISTA(prob, np.kron(W, np.eye(d)), T, initial_lambda=1e-1, initial_gamma=1)#/(1.01*np.linalg.norm(prob.A_s, 2)**2) )#/(1.01*np.linalg.norm(W.T@prob.A_s, 2)))

start = time.time()
training_stages = train.setup_training(layers,prob,trinit=1e-3)
end = time.time()
print( 'Took me {totaltime:.3f} minutes for setup training'.format(totaltime = (end-start)/60))

# Train!
# 
sess, nmse_history, loss_history, nmse_switch  = train.do_training(training_stages,prob,'train_'+str(m)+'/pnz_001_Gauss_ALBISTA_case_'+str(case)+'_T'+str(T)+'_SNRdB_'+str(SNR)+'batch'+str(MC)+'.npz')
#sess = train.do_training(training_stages,prob,'LBISTA-TiLBISTA_case_'+str(case)+'_T'+str(T)+'_SNRdB_'+str(SNR)+'batch'+str(MC)+'.npz')

############################ without training ########################
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# nmse_history = []
# loss_history = []
# nmse_switch = []
##################################################################


y,x = prob(sess)
MC = x.shape[-1]
t=0
l2norm=np.zeros(((T),MC))
nmse_dbLISTA=np.zeros(((T),MC))
for name, xhat_, var_list in layers:
    if not name=='Linear':
        xhat = sess.run(xhat_, feed_dict={prob.y_: y, prob.x_: x})
        for i in range(0, x.shape[1]):
            nmse_dbLISTA[t,i]=bst.nmse(xhat[:,i, np.newaxis], x[:,i, np.newaxis])
            l2norm[t, i] = bst.l21norm(xhat[:, i]- x[:, i], prob.L, prob.B)
        t+=1

nmse_dbLISTAMean = 10*np.log10(np.mean(np.ma.masked_invalid(nmse_dbLISTA), axis=1))
l2normLISTAMean = np.mean(np.ma.masked_invalid(l2norm), axis=1)
l2normmax = np.max(l2norm, axis=1)
lam = np.zeros(T)
gam = np.zeros(T)
k = 1
for name, xhat_, var_list in layers:
    if not name == 'Linear':
        lam[k-1], gam[k-1] = sess.run([layers[k-1][2][0], layers[k-1][2][1]])
        k = k+1


sio.savemat('mat_'+str(m)+'/Gauss_ALBISTA_case_'+str(case)+'_T'+str(T)+'_SNRdB_'+str(SNR)+'batch'+str(MC), {'nmse_dbLISTAMean': nmse_dbLISTAMean, 'l2normLISTAMean': l2normLISTAMean, 'l2normmax': l2normmax, 'nmse_history': nmse_history, 'loss_history': loss_history, 'nmse_switch': nmse_switch})
print(nmse_dbLISTAMean)

sio.savemat('mat_'+str(m)+'/Gausssave_param_ALBISTA_case_'+str(case)+'_T'+str(T)+'_SNRdB_'+str(SNR)+'batch'+str(MC)+'mat', {'nmse_dbLISTAMean': nmse_dbLISTAMean, 'l2normLISTAMean': l2normLISTAMean, 'l2normmax': l2normmax, 'lam': lam, 'gam': gam})