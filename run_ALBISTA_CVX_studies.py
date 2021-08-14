from __future__ import division
from __future__ import print_function
from __future__ import division
from __future__ import print_function
import numpy as np
SNR_list=[np.infty,40, 20, 10]
for k in range(0, len(SNR_list)):

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

    T=16
    # SNR = np.inf #wird SNR_dB = -10 mein Gegenbeispiel?
    SNR = SNR_list[k]
    MC = 250
    prob = problem.block_gaussian_trial(m=50, L=75, B=1, MC=MC, pnz=0.1, SNR_dB=SNR) # a Block-Gaussian x, noisily observed through a random matrix
    # pdb.set_trace()
    # layers, W = network.build_BALISTA(prob,T=T,initial_lambda=0.1, initial_gamma=2/(1.01*la.norm(prob.A,2)**2), untied=False)
    layers, W = network.build_BALISTA_v5(prob,T=T,initial_lambda=0.5, initial_gamma=0.1, untied=False)
    #layers = network.build_TiBLISTA(prob,T=T,initial_lambda=0.1, initial_gamma=1, untied=False)
    # pdb.set_trace()
    start = time.time()
    training_stages = train.setup_training(layers,prob,trinit=1e-3,refinements=(0.5, 0.01, 0.001))
    end = time.time()
    print( 'Took me {totaltime:.3f} minutes for setup training'.format(totaltime = (end-start)/60))

    # sess = train.do_training(training_stages,prob,'TiBLISTA_block_Gauss_giidT12Thermo6-Jan.npz')
    sess = train.do_training(training_stages,prob,'trainings/batch250_sparse_case_D_ALBISTA_CVX_v4_block_Gauss_giidT'+str(T)+'Thermo6-JanWithSNR_dB'+str(SNR)+'.npz')
    """# Evaluating"""
    # sparsemat = sio.loadmat('blocktest.mat')
    # y = sparsemat.get('y')
    # x = sparsemat.get('x')
    # y,x = prob(sess)
    sparsemat = sio.loadmat('mat/sparse_case_D_blocktestSNR' + str(SNR) + '.mat')
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
    mut_block = bst.mutual_block_coherence(W,prob.A, prob.L, prob.B)
    Cw = bst.C_w(W, prob.L, prob.B)
    C=np.max(gam)*Cw
    sigma = np.linalg.norm(prob.noise, 2)
    #sio.savemat('mat/BALISTA_v4_T'+str(T)+'WithSNR_dB'+str(SNR)+'.mat', {'xhat': xhat, 'lam': lam, 'gam': gam, 'x': x, 'y': y, 'l21norm': l2norm, 'nmse': nmse_dbLISTA, 'noise': prob.noise})
    # plt.plot(10*np.log10((lam)/gam), label = 'lam/gam')
    # plt.plot(10*np.log10(l2normmax), label = 'l21 max')
    # plt.legend()
    # plt.title('sigma = ' + str(sigma))
    # plt.savefig('images/BALISTA_v4Thm6T'+str(T)+'WithSNR_dB'+str(SNR)+'.png')
    # plt.close()
    # plt.plot(gam)
    # plt.title('sigma = ' + str(sigma))
    # plt.savefig('images/BALISTA_v4GammasT'+str(T)+'WithSNR_dB'+str(SNR)+'.png')
    # plt.close()
    # plt.plot(10*np.log10(nmse_dbLISTAMean), label = 'nmse_db mean')
    # plt.legend()
    # plt.title('sigma = ' + str(sigma))
    # plt.savefig('images/BALISTA_v4MeanNmseDBT'+str(T)+'WithSNR_dB'+str(SNR)+'.png')
    # plt.close()
    # plt.plot(lam, label = 'lambdas')
    # plt.legend()
    # plt.title('sigma = ' + str(sigma))
    # plt.savefig('images/BALISTA_v4lambdas'+str(T)+'WithSNR_dB'+str(SNR)+'.png')
    # plt.close()
    #pdb.set_trace()
    tf.reset_default_graph()
    sess.close()
    #fig, axes = plt.subplots(1,2)
    #axes[0].plot(xhat[:,10])
    #axes[1].plot(x[:,10])
    # pdb.set_trace()
    sio.savemat('mat/sparse_case_D_ALBISTA_CVXnmseSNR' + str(SNR) + 'dB.mat', {'nmseMean': nmse_dbLISTAMean, 'nmse': nmse_dbLISTA, 'l21norm': l2norm})
    sio.savemat('mat/sparse_case_D_ALBISTA_CVXalpha.mat', {'al': lam, 'gam': gam})
    #sio.savemat('lossfunction/testxySNR'+str(SNR)+'dB.mat', {'x': x, 'y':y, 'xhat_BALISTA': xhat})
    tf.reset_default_graph()
    sess.close()
    #
    for name in dir():
        if not (name.startswith('SNR_list') or name.startswith('omega_list')):
            del globals()[name]
