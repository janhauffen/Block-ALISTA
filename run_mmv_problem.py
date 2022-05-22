# -*- coding: utf-8 -*-
"""run_mmv_problem_case10.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1PPPsSZ4AYKYq6bo1aOd2-DSFgPIKSy40
"""
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
import problem, network_mmv, train
import blocksparsetoolbox as bst

from scipy.linalg import toeplitz, circulant

np.random.seed(1)
tf.set_random_seed(1)

# Creating the mmv-problem: select circular matrix case 1-8
def R(D, n, d):
    R = np.zeros((n,n))
    for k in range(0,n):
        for l in range(0,n):
            if k != l:
                R[k,l] = np.linalg.norm(D[:, k*d:(k+1)*d].T@D[:, l*d:(l+1)*d], 2)
    return R
#case = 10 #ohne _ vor namen
#n = 128
#m = 20
#d = 15
#r = 5

case = 10 
n = 128
#m = 48 
m = 32
d = 15
K = np.zeros((m,n))
for l in range(0,n):
        K[:,l:(l+1)] = np.random.random(size=(m, 1))
        K[:,l:(l+1)], _ = np.linalg.qr(K[:,l:(l+1)])

print('rank(D^T@D)='+str(np.linalg.matrix_rank(K.T@K)))
print('rank(D@D^T)='+str(np.linalg.matrix_rank(K@K.T)))
print('rank(D)='+str(np.linalg.matrix_rank(K)))
print('coherence = '+str(np.max(np.abs(K.T@K-np.eye(n)))))

D_ = np.kron(K, np.eye(d))
mu_b = 1/d*np.max(R(D_, n, d))
print('block-coherence: ' + str(mu_b))
print('block sparsity bound = ' + str((1/mu_b+d)/(2*d)))

SNR = 20 # signal to noise ratio given in dB
MC = 250 # batch number
prob = problem.mmv_problem(K,B=d, MC=MC, pnz=0.1, SNR_dB=SNR) #case 5 with SNR = 30dB and pnz = 0.01 works. 
                                                                   #Be careful here, because poor coherence does not allow for great sparsity

# creating the network and setup training:

T = 6 # number of layers/iterations

# computing the analytical weight matrix for ALBISTA and NA-ALBISTA

W = bst.compute_W_Lagrange(prob.A_s, prob.L, 1)

layers = network_mmv.build_CircALBISTA(prob, np.kron(W ,np.eye(d)), T, initial_lambda=1e-1, initial_gamma=1)#/(1.01*np.linalg.norm(W.T@prob.A_s, 2)))
#layers = network_mmv.build_TiBLISTA(prob, prob.A_s, T, initial_lambda=1e-1, initial_gamma=2/ (1.01*np.linalg.norm(prob.A_s,2)**2))

#layers = network_mmv.build_CircALBISTA(prob, np.kron(W[:,0:m+1],np.eye(d)), T, initial_lambda=1e-1, initial_gamma=1/(np.linalg.norm(prob.A_s)))
#layers = network_mmv.build_TiBLISTA(prob, T, initial_lambda=1e-1, initial_gamma=1/(np.linalg.norm(prob.A_s)))

start = time.time()
training_stages = train.setup_training(layers,prob,trinit=1e-3)
end = time.time()
print( 'Took me {totaltime:.3f} minutes for setup training'.format(totaltime = (end-start)/60))

# Train!
# 
sess, nmse_history, loss_history, nmse_switch  = train.do_training(training_stages,prob,'train_'+str(m)+'/ALBISTA_case_'+str(case)+'_T'+str(T)+'_SNRdB_'+str(SNR)+'batch'+str(MC)+'.npz')
#sess = train.do_training(training_stages,prob,'LBISTA-TiLBISTA_case_'+str(case)+'_T'+str(T)+'_SNRdB_'+str(SNR)+'batch'+str(MC)+'.npz')

##################### nur fuer orig BISTA########################
#sess = tf.Session()
#sess.run(tf.global_variables_initializer())
#nmse_history = []
#loss_history = []
#nmse_switch = []
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

#fig = plt.figure(figsize=(6, 6), dpi=150)
#plt.plot(np.log(l2normmax))
#plt.plot(np.log(lam/gam))
#plt.plot(W[:,0])
#plt.savefig('ALBISTA_Thm_T'+str(T)+'.png')
#sio.savemat('mat_2/proposed_gamma_ALBISTA_case_'+str(case)+'_T'+str(T)+'_SNRdB_'+str(SNR)+'batch'+str(MC), {'nmse_dbLISTAMean': nmse_dbLISTAMean, 'l2normLISTAMean': l2normLISTAMean, 'l2normmax': l2normmax, 'nmse_history': nmse_history, 'loss_history': loss_history, 'nmse_switch': nmse_switch})
#print(nmse_dbLISTAMean)

sio.savemat('mat_'+str(m)+'/save_param_ALBISTA_case_'+str(case)+'_T'+str(T)+'_SNRdB_'+str(SNR)+'batch'+str(MC)+'mat', {'nmse_dbLISTAMean': nmse_dbLISTAMean, 'l2normLISTAMean': l2normLISTAMean, 'l2normmax': l2normmax, 'lam': lam, 'gam': gam})