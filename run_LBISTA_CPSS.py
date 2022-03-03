from __future__ import division
from __future__ import print_function
from __future__ import division
from __future__ import print_function
import numpy as np
import os
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
np.random.seed(1) # numpy is good about making repeatable output
tf.set_random_seed(1) # on the other hand, this is cbasically useless (see issue 9171)

SNR_list=[np.infty, 20]
k = 1
T=16
SNR = SNR_list[k]
MC = 250
prob = problem.block_gaussian_trial(m=50, L=15, B=5, MC=MC, pnz=0.1, SNR_dB=SNR)  # a Block-Gaussian x, noisily observed through a random matrix
# pdb.set_trace()
# layers, W = network.build_BALISTA(prob,T=T,initial_lambda=0.1, initial_gamma=2/(1.01*la.norm(prob.A,2)**2), untied=False)
layers = network.build_LBISTA_CPSS(prob,T=T,initial_lambda=0.1, initial_gamma=0.1, untied=False)
#layers = network.build_TiBLISTA(prob,T=T,initial_lambda=0.1, initial_gamma=1, untied=False)
# pdb.set_trace()
start = time.time()
training_stages = train.setup_training(layers,prob,trinit=1e-3,refinements=(0.5, 0.01, 0.001))
end = time.time()
print( 'Took me {totaltime:.3f} minutes for setup training'.format(totaltime = (end-start)/60))

# sess = train.do_training(training_stages,prob,'TiBLISTA_block_Gauss_giidT12Thermo6-Jan.npz')
sess = train.do_training(training_stages,prob,'trainings/LBISTA_CPSS_T'+str(T)+'_SNRdB_'+str(SNR)+'batch'+str(MC)+'.npz')
"""# Evaluating"""
# sparsemat = sio.loadmat('mat/normalized_D_blocktestSNR' + str(SNR) + '.mat')
# y = sparsemat.get('y')
# x = sparsemat.get('x')
#y,x = prob(sess)
#sio.savemat('mat/blocktestSNR'+str(SNR)+'.mat', {'y': y, 'x':x, 'D': prob.A})
sparsemat = sio.loadmat('mat/small_normalized_D_blocktestSNR' + str(SNR) + '.mat')
y = sparsemat.get('y')
x = sparsemat.get('x')
MC=x.shape[-1]

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

nmse_dbLISTAMean = np.mean(np.ma.masked_invalid(nmse_dbLISTA), axis=1)
l2normLISTAMean = np.mean(np.ma.masked_invalid(l2norm), axis=1)
l2normmax = np.max(l2norm, axis=1)

lam = np.zeros(T)
gam = np.zeros(T)
k = 1
for name, xhat_, var_list in layers:
    if not name == 'Linear':
        lam[k-1], gam[k-1] = sess.run([layers[k-1][2][0], layers[k-1][2][1]])
        k = k+1

#s = bst.sparsity(x[:,10], 16, 8)
#mut_block = bst.mutual_block_coherence(W,prob.A, prob.L, prob.B)
#Cw = bst.C_w(W, prob.L, prob.B)
#C=np.max(gam)*Cw
sigma = np.linalg.norm(prob.noise, 2)
#sio.savemat('mat/BALISTA_v4_T'+str(T)+'WithSNR_dB'+str(SNR)+'.mat', {'xhat': xhat, 'lam': lam, 'gam': gam, 'x': x, 'y': y, 'l21norm': l2norm, 'nmse': nmse_dbLISTA, 'noise': prob.noise})
#plt.plot(10*np.log10((lam)/gam), label = 'lam/gam')
#plt.plot(10*np.log10(l2normmax), label = 'l21 max')
#plt.legend()
#plt.title('sigma = ' + str(sigma))
#plt.savefig('images/LBISTA_CPSSThm6T'+str(T)+'WithSNR_dB'+str(SNR)+'.png')
#plt.close()
#plt.plot(gam)
#plt.title('sigma = ' + str(sigma))
#plt.savefig('images/LBISTA_CPSSGammasT'+str(T)+'WithSNR_dB'+str(SNR)+'.png')
#plt.close()
#plt.plot(10*np.log10(nmse_dbLISTAMean), label = 'nmse_db mean')
#plt.legend()
#plt.title('sigma = ' + str(sigma))
#plt.savefig('images/LBISTA_CPSSMeanNmseDBT'+str(T)+'WithSNR_dB'+str(SNR)+'.png')
#plt.close()
#plt.plot(lam, label = 'lambdas')
#plt.legend()
#plt.title('sigma = ' + str(sigma))
#plt.savefig('images/LBISTA_CPSSlambdas'+str(T)+'WithSNR_dB'+str(SNR)+'.png')
#plt.close()
sio.savemat('mat/normalized_D_LBISTA_CPSSnmseSNR'+str(SNR)+'dB.mat', {'nmseMean': nmse_dbLISTAMean, 'nmse': nmse_dbLISTA})
sio.savemat('mat/normalized_D_LBISTA_CPSSalpha.mat', {'al': lam, 'gam': gam})

