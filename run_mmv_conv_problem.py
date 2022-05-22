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
import problem, network_mmv_conv, train
import blocksparsetoolbox as bst

from scipy.linalg import toeplitz, circulant

np.random.seed(1)
tf.set_random_seed(1)

# Creating the mmv-problem: select circular matrix case 1-8

case = 12
n = 128
m = n
d = 15
#r = 12 (case = 10)
#r = 16 #(case = 11)
r = 19

T = np.zeros((n,n))
T[0,-1]=1
T[1:n,0:n-1]=np.eye(n-1)

if case==1:
  #1 Example: Random Circular Symmetric Matrix, where D^T@D has full rank
  a = np.random.normal(size=(m))
  D = circulant(a).astype('float32')
  D = 1/2*(D+D.T)
  D = 1/np.linalg.norm(D[:,0])*D
  for i in range(1,n):
    D[:,i]=T@D[:,i-1]

  print('rank(D^T@D)='+str(np.linalg.matrix_rank(D.T@D)))
  print('rank(D)='+str(np.linalg.matrix_rank(D)))
  print('coherence = '+str(np.max(np.abs(D.T@D-np.eye(m)))))
elif case==2:
  #2 Example: Symmetric Circular Matrix, based upon discrete function values, where D^T@D has low rank

  x = np.linspace(-1,1,n)
  a = 0.5-2*x**2
  D = circulant(a).astype('float32')
  D = 1/np.linalg.norm(D[:,0])*D

  print('rank(D^T@D)='+str(np.linalg.matrix_rank(D.T@D)))
  print('rank(D)='+str(np.linalg.matrix_rank(D)))
  print('coherence = '+str(np.max(np.abs(D.T@D-np.eye(m)))))
elif case==3:
  #3 Example: Symmetric Circular Matrix, based upon discrete function values, where D^T@D has low rank

  x = np.linspace(-1,1,n)
  a = (np.cos(1*np.pi*x))
  #a = 1/np.linalg.norm(a)*np.abs(a) #no problem
  a = 1/np.linalg.norm(a)*(a) #problem
  D = circulant(a).astype('float32')

  print('rank(D^T@D)='+str(np.linalg.matrix_rank(D.T@D)))
  print('rank(D)='+str(np.linalg.matrix_rank(D)))
  print('coherence = '+str(np.max(np.abs(D.T@D-np.eye(m)))))
elif case==4:
  #4 Example: Random Circular Matrix

  a = np.random.normal(size=(m))
  D = circulant(a).astype('float32')
  D = 1/np.linalg.norm(D[:,0])*D

  print('rank(D^T@D)='+str(np.linalg.matrix_rank(D.T@D)))
  print('rank(D@D^T)='+str(np.linalg.matrix_rank(D@D.T)))
  print('rank(D)='+str(np.linalg.matrix_rank(D)))
  print('coherence = '+str(np.max(np.abs(D.T@D-np.eye(m)))))
elif case==5:
  #5 Example: Circular Matrix, based upon discrete function values,
  
  x = np.linspace(-1,1,n)
  a = (np.sin(1*np.pi*x))
  #a = 1/np.linalg.norm(a)*np.abs(a) #no problem
  a = 1/np.linalg.norm(a)*(a) #problem
  D = circulant(a).astype('float32')

  print('rank(D^T@D)='+str(np.linalg.matrix_rank(D.T@D)))
  print('rank(D@D^T)='+str(np.linalg.matrix_rank(D@D.T)))
  print('rank(D)='+str(np.linalg.matrix_rank(D)))
  print('coherence = '+str(np.max(np.abs(D.T@D-np.eye(m)))))
elif case==6:
  #6 Example: Circular Matrix, based upon discrete function values,
  
  x = np.linspace(-1,1,n)
  a = (np.exp(1*np.pi*x))
  #a = 1/np.linalg.norm(a)*np.abs(a) #no problem
  a = 1/np.linalg.norm(a)*(a) #problem
  D = circulant(a).astype('float32')

  print('rank(D^T@D)='+str(np.linalg.matrix_rank(D.T@D)))
  print('rank(D@D^T)='+str(np.linalg.matrix_rank(D@D.T)))
  print('rank(D)='+str(np.linalg.matrix_rank(D)))
  print('coherence = '+str(np.max(np.abs(D.T@D-np.eye(m)))))
elif case==7:
  #7 Example: Circular Matrix, not symmetric but low rank
  x = np.linspace(-1,1,n)
  a = np.cos(np.pi*x)
  a[30:n]=0
  D = circulant(a).astype('float32')
  D = 1/np.linalg.norm(D[:,0])*D

  print('rank(D^T@D)='+str(np.linalg.matrix_rank(D.T@D))) 
  print('rank(D@D^T)='+str(np.linalg.matrix_rank(D@D.T)))
  print('rank(D)='+str(np.linalg.matrix_rank(D)))
  print('coherence = '+str(np.max(np.abs(D.T@D-np.eye(m)))))
elif case==8:
  #8 Example: Circular Matrix, 
  x = np.linspace(-1,1,n)
  a = np.exp(np.pi*x)
  a[30:n]=a[0:30]
  D = circulant(a).astype('float32')
  D = 1/np.linalg.norm(D[:,0])*D

  print('rank(D^T@D)='+str(np.linalg.matrix_rank(D.T@D))) 
  print('rank(D@D^T)='+str(np.linalg.matrix_rank(D@D.T)))
  print('rank(D)='+str(np.linalg.matrix_rank(D)))
  print('coherence = '+str(np.max(np.abs(D.T@D-np.eye(m)))))
elif case==9:
  #9 Toeplitz Case:
  x = np.linspace(-1,1,n-30)
  a = np.random.normal(size=(n-30))
  padding = np.zeros(np.round(n), a.dtype)
  first_col = np.r_[a, padding]
  first_row = np.r_[a[0], padding]
  D = toeplitz(first_col, first_row)
  D = 1/np.linalg.norm(D[:,0])*D
  n,m = D.shape
  print('rank(D^T@D)='+str(np.linalg.matrix_rank(D.T@D))) 
  print('rank(D@D^T)='+str(np.linalg.matrix_rank(D@D.T)))
  print('rank(D)='+str(np.linalg.matrix_rank(D)))
  print('coherence = '+str(np.max(np.abs(D.T@D-np.eye(m+1)))))
elif case==10:
  k = np.zeros(n)
  a = np.random.normal(size=(r,2))#.view(np.complex128)  #komplex erzeugen!
  s_i = np.random.permutation(n)
  s = s_i[0:r]
  k[s] = a[:,0] #oder hier symmetrisch <-
  k = 1/2*(k+k[-np.arange(-n+1,1)])
  k_s = k
  k = n*np.real(np.fft.ifft(k))
  D = circulant(k).astype('float32')
  D = 1/np.linalg.norm(D[:,0])*D
  print('rank(D^T@D)='+str(np.linalg.matrix_rank(D.T@D))) 
  print('rank(D@D^T)='+str(np.linalg.matrix_rank(D@D.T)))
  print('rank(D)='+str(np.linalg.matrix_rank(D)))
  print('coherence = '+str(np.max(np.abs(D.T@D-np.eye(m)))))
elif case==11:
  #4 Example: Random Circular Matrix
  k = np.zeros(n)
  a = np.random.normal(size=(r,2))
  s_i = np.random.permutation(n)
  s = s_i[0:r]
  k[s] = a[:,0] 
  #k = 1/2*(k+k[-np.arange(-n+1,1)])
  k_s = k
  k = n*np.real(np.fft.ifft(k))
  D = circulant(k).astype('float32')
  D = 1/np.linalg.norm(D[:,0])*D
  print('rank(D^T@D)='+str(np.linalg.matrix_rank(D.T@D))) 
  print('rank(D@D^T)='+str(np.linalg.matrix_rank(D@D.T)))
  print('rank(D)='+str(np.linalg.matrix_rank(D)))
  print('coherence = '+str(np.max(np.abs(D.T@D-np.eye(m)))))
elif case == 12:
  # Triangle
  h = np.zeros(n)
  h[0:int(r/2)] = 1-np.linspace(0,1 , int(r/2))
  h[int(-r/2+1):n] = np.linspace(0,h[1] , int(r/2)-1)
  #h = (h+h[-np.arange(-n+1,1)])
  k = (np.fft.ifft(h))
  D = circulant(k).astype('float32')
  D = 1/np.linalg.norm(D[:,0], 2)*D
  n = D.shape[0]
  print('rank(D^T@D)='+str(np.linalg.matrix_rank(D.T@D))) 
  print('rank(D@D^T)='+str(np.linalg.matrix_rank(D@D.T)))
  print('rank(D)='+str(np.linalg.matrix_rank(D)))
  print('coherence = '+str(np.max(np.abs(D.T@D-np.eye(n)))))

def R(D, n, d):
    R = np.zeros((n,n))
    for k in range(0,n):
        for l in range(0,n):
            if k != l:
                R[k,l] = np.linalg.norm(D[:, k*d:(k+1)*d].T@D[:, l*d:(l+1)*d], 2)
    return R
D_ = np.kron(D, np.eye(d))
mu_b = 1/d*np.max(R(D_, n, d))
print('block-coherence: ' + str(mu_b))
print('block sparsity bound = ' + str((1/mu_b+d)/(2*d)))

SNR = np.infty # signal to noise ratio given in dB
MC = 250 # batch number
prob = problem.mmv_problem(D,B=d, MC=MC, pnz=0.1, SNR_dB=SNR)
                                                                   #Be careful here, because poor coherence does not allow for great sparsity


# creating the network and setup training:

T = 6 # number of layers/iterations

# computing the analytical weight matrix for ALBISTA and NA-ALBISTA

W = network_mmv_conv(circulant(prob.A_s[:,0])) # if A_s represents a circular convolution we can easily compute the analytical 
#

layers = network_mmv_conv(prob, np.kron(W ,np.eye(d)), T, initial_lambda=.1, initial_gamma=1)#/(1.01*np.linalg.norm(W.T@prob.A_s, 2)))
#layers = network_mmv_conv(prob, W, T, initial_lambda=1e-1, initial_gamma=1)
#layers = network_mmv_conv(prob, W, T, initial_lambda=1e-1)#, initial_gamma=1)
#layers = network_mmv_conv(prob, np.kron(W[:,0:m+1],np.eye(d)), T, initial_lambda=1e-1, initial_gamma=1/(np.linalg.norm(prob.A_s)))
#layers = network_mmv_conv(prob, T, initial_lambda=1e-1, initial_gamma=1/(np.linalg.norm(prob.A_s)))

start = time.time()
training_stages = train.setup_training(layers,prob,trinit=1e-3)
end = time.time()
print( 'Took me {totaltime:.3f} minutes for setup training'.format(totaltime = (end-start)/60))

# Train!
#LBISTA/Experiments/Second Experiments/MMV_conv/mat_10/ALBISTAvsALBISTA/r_1/proposed_gamma_ALBISTA_case_11_T6_SNRdB_infbatch250
sess, nmse_history, loss_history, nmse_switch  = train.do_training(training_stages,prob,'train_r_'+str(r)+'/gamma_1_ALBISTA_case_'+str(case)+'_T'+str(T)+'_SNRdB_'+str(SNR)+'batch'+str(MC)+'.npz')
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
        
#b = sess.run(layers[-1][2][2], feed_dict={prob.y_: y, prob.x_: x})
nmse_dbLISTAMean = 10*np.log10(np.mean(np.ma.masked_invalid(nmse_dbLISTA), axis=1))
l2normLISTAMean = np.mean(np.ma.masked_invalid(l2norm), axis=1)
l2normmax = np.max(l2norm, axis=1)

lam = np.zeros(T)
gam = np.zeros(T)
#k = 1
#for name, xhat_, var_list in layers:
#    if not name == 'Linear':
#        lam[k-1], gam[k-1] = sess.run([layers[k-1][2][0], layers[k-1][2][1]])
#        k = k+1

#fig = plt.figure(figsize=(6, 6), dpi=150)
#plt.plot(np.log(l2normmax))
#plt.plot(np.log(lam/gam))
#plt.plot(W[:,0])
#plt.savefig('ALBISTA_Thm_T'+str(T)+'.png')
sio.savemat('mat_r_'+str(r)+'/gamma_1_ALBISTA_case_'+str(case)+'_T'+str(T)+'_SNRdB_'+str(SNR)+'batch'+str(MC), {'nmse_dbLISTAMean': nmse_dbLISTAMean, 'l2normLISTAMean': l2normLISTAMean, 'l2normmax': l2normmax, 'nmse_history': nmse_history, 'loss_history': loss_history, 'nmse_switch': nmse_switch})
#print(nmse_dbLISTAMean)

#sio.savemat('mat_r_'+str(r)+'/tiedLBISTA_case_'+str(case)+'_T'+str(T)+'_SNRdB_'+str(SNR)+'batch'+str(MC)+'mat', {'nmse_dbLISTAMean': nmse_dbLISTAMean, 'l2normLISTAMean': l2normLISTAMean, 'l2normmax': l2normmax, 'lam': lam, 'gam': gam})