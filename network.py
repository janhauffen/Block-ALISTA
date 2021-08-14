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
import blocksparsetoolbox as bst
import keras.backend as K
import cvxpy as cp

def R(D, n, d): #function to compute the generalized coherence for D = B^TD
    n_y, n_x = D.shape
    R = np.zeros((n, n))
    for k in range(0, n):
        for l in range(k, n):
            I = np.zeros((n_x, n_x))
            I_s = np.zeros((n, n))
            I[l * d:(l + 1) * d, k * d:(k + 1) * d] = np.ones(d)
            I_s[l, k] = 1
            if k == l:
                R = R + cp.norm(cp.multiply(I, D), 2) * I_s
            else:
                R = R + cp.norm(cp.multiply(I, D), 2) * I_s + cp.norm(cp.multiply(I.T, D), 2) * I_s.T

    return R

def simple_soft_threshold(r_, lam_):
    "implement a soft threshold function y=sign(r)*max(0,abs(r)-lam)"
    lam_ = tf.maximum(lam_, 0)
    #pdb.set_trace()
    return tf.sign(r_) * tf.maximum(tf.abs(r_) - lam_, 0)

def block_soft_threshold(X_, al_):
    B = 15
    L = 5
    shape = K.shape(X_)
    pool_shape1 = tf.stack([B, L, shape[1]])
    al_ = tf.maximum(al_, 0)
    Xnew_ = K.reshape(X_, pool_shape1)  # i'th coloumn of X_
    r_ = tf.sqrt(tf.reduce_sum((Xnew_ ** 2), 1))
    r_ = tf.maximum(.0, 1 - tf.math.divide_no_nan(al_ , r_))
    shaper = K.shape(r_)
    pool_r = tf.stack([B, 1, shaper[1]])
    r_ = K.reshape(r_, pool_r)  # Diesen reshape Befehl auch zu K.reshape ändern
    pool_shape2 = tf.stack([B * L, shape[1]])

    Xnew_ = tf.multiply(Xnew_, r_)

    return K.reshape(Xnew_, pool_shape2)  # transpose, bc of the list and tf.stack() operator

def block_soft_threshold_elastic_net(X_, al_, la_):
    B = 64
    L = 1

    shape = K.shape(X_)
    pool_shape1 = tf.stack([B, L, shape[1]])
    al_ = tf.maximum(al_, 0)
    Xnew_ = K.reshape(X_, pool_shape1)  # i'th coloumn of X_
    r_ = tf.sqrt(tf.reduce_sum((Xnew_ ** 2), 1))
    r_ = tf.maximum(.0, 1 - tf.math.divide_no_nan(al_ , r_))
    shaper = K.shape(r_)
    pool_r = tf.stack([B, 1, shaper[1]])
    r_ = K.reshape(r_, pool_r)  # Diesen reshape Befehl auch zu K.reshape ändern
    pool_shape2 = tf.stack([B * L, shape[1]])

    Xnew_ = tf.multiply(Xnew_, r_)*(1+la_)**(-1)

    return K.reshape(Xnew_, pool_shape2)  # transpose, bc of the list and tf.stack() operator

def build_LBISTA(prob,T,initial_lambda=.1,untied=False):
    """
    Builds a LISTA network to infer x from prob.y_ = matmul(prob.A,x) + AWGN
    prob            - is a TFGenerator which contains problem parameters and def of how to generate training data
    initial_lambda  - could be some parameter of Block ISTA <- DELETE if unnecessary
    untied          - flag for tied or untied case
    Return a list of layer info (name,xhat_,newvars)
     name : description, e.g. 'LISTA T=1'
     xhat_ : that which approximates x_ at some point in the algorithm
     newvars : a tuple of layer-specific trainable variables
    """
    blocksoft=block_soft_threshold
    layers = []
    A = prob.A
    M,N = A.shape
    B = A.T / (1.01 * la.norm(A,2)**2)
    B_ =  tf.Variable(B,dtype=tf.float32,name='B_0')
    By_ = tf.matmul(B_,prob.y_)
    S_ = tf.Variable( np.identity(N) - np.matmul(B,A),dtype=tf.float32,name='S_0')
    #pdb.set_trace()
    #layers.append( ('Linear',By_,None) )
    
    initial_lambda = np.array(initial_lambda).astype(np.float32)
    #if getattr(prob,'iid',True) == False:
    #    # create a parameter for each coordinate in x
    #    initial_lambda = initial_lambda*np.ones( (N,1),dtype=np.float32 )
    lam0_ = tf.Variable( initial_lambda,name='lam_0')
    xhat_ = blocksoft( By_, lam0_)
    layers.append( ('LBISTA T=1',xhat_, (lam0_,B_, S_) ) )
    #pdb.set_trace()
    for t in range(1,T):
        lam_ = tf.Variable( initial_lambda,name='lam_{0}'.format(t) )
        xhat_ = blocksoft( tf.matmul(S_,xhat_) + tf.matmul(B_,prob.y_), lam_ )
        layers.append( ('LBISTA T='+str(t+1),xhat_,(lam_,B_, S_)) )

    """
    # check other functions in this file (e.g., build_LISTA and build_LAMP4SSC) to implement LBISTA network
    # send me questions if needed
    """

    return layers


def build_LBISTA_untied(prob,T,initial_lambda=.1,untied=False):
    """
    Builds a LISTA network to infer x from prob.y_ = matmul(prob.A,x) + AWGN
    prob            - is a TFGenerator which contains problem parameters and def of how to generate training data
    initial_lambda  - could be some parameter of Block ISTA <- DELETE if unnecessary
    untied          - flag for tied or untied case
    Return a list of layer info (name,xhat_,newvars)
     name : description, e.g. 'LISTA T=1'
     xhat_ : that which approximates x_ at some point in the algorithm

     newvars : a tuple of layer-specific trainable variables
    """
    blocksoft=block_soft_threshold
    layers = []
    A = prob.A
    M,N = A.shape
    B = A.T / (1.01 * la.norm(A,2)**2)
    B_ =  tf.Variable(B,dtype=tf.float32,name='B_0')
    S_ = tf.Variable( np.identity(N) - np.matmul(B,A),dtype=tf.float32,name='S_0')
    By_ = tf.matmul(B_,prob.y_)
    #pdb.set_trace()
    #layers.append( ('Linear',By_,None) )
    
    initial_lambda = np.array(initial_lambda).astype(np.float32)
    #if getattr(prob,'iid',True) == False:
    #    # create a parameter for each coordinate in x
    #    initial_lambda = initial_lambda*np.ones( (N,1),dtype=np.float32 )
    lam0_ = tf.Variable( initial_lambda,name='lam_0')
    xhat_ = blocksoft( By_, lam0_)
    layers.append( ('LBISTA T=1',xhat_, (lam0_,B_, S_) ) )
    #pdb.set_trace()
    for t in range(1,T):
        B_ =  tf.Variable(B,dtype=tf.float32,name='B_{0}'.format(t) )
        S_ = tf.Variable( np.identity(N) - np.matmul(B,A),dtype=tf.float32,name='S_{0}'.format(t) )
        lam_ = tf.Variable( initial_lambda,name='lam_{0}'.format(t) )
        xhat_ = blocksoft( tf.matmul(S_,xhat_) + tf.matmul(B_,prob.y_), lam_ )
        layers.append( ('LBISTA T='+str(t+1),xhat_,(lam_,B_, S_)) )

    """

    # check other functions in this file (e.g., build_LISTA and build_LAMP4SSC) to implement LBISTA network
    # send me questions if needed
    """

    return layers

def build_ALISTA(prob, T, initial_lambda=.1, initial_gamma=1, untied=False):
    """
    Builds a LISTA network to infer x from prob.y_ = matmul(prob.A,x) + AWGN
    prob            - is a TFGenerator which contains problem parameters and def of how to generate training data
    initial_lambda  - could be some parameter of Block ISTA <- DELETE if unnecessary
    untied          - flag for tied or untied case
    Return a list of layer info (name,xhat_,newvars)
     name : description, e.g. 'LISTA T=1'
     xhat_ : that which approximates x_ at some point in the algorithm
     newvars : a tuple of layer-specific trainable variables
    """
    blocksoft = simple_soft_threshold
    layers = []
    A = prob.A
    M, N = A.shape
    Wmat = sio.loadmat('W.mat') #W has to be precomputed before running the script
    W = Wmat.get('W')
    gamma = initial_gamma
    gamma0_ = tf.Variable(gamma, dtype=tf.float32, name='gamma_0')
    W_ = tf.Variable(np.transpose(W), dtype=tf.float32, trainable = False)
    layers.append(('Linear', tf.matmul(gamma0_*W_, prob.y_), None))
    initial_lambda = np.array(initial_lambda).astype(np.float32)
    lam0_ = tf.Variable(initial_lambda, name='lam_0')
    xhat_ = blocksoft(tf.matmul(gamma0_*W_, prob.y_), lam0_)
    layers.append(('LBISTA T=1', xhat_, (lam0_,gamma0_)))
    # pdb.set_trace()
    for t in range(1, T):
        lam_ = tf.Variable(initial_lambda, name='lam_{0}'.format(t))
        gamma_ = tf.Variable(gamma, dtype=tf.float32, name='gamma_{0}'.format(t))
        xhat_ = blocksoft(xhat_ - tf.matmul(gamma_*W_, tf.matmul(prob.A_, xhat_) - prob.y_), lam_)
        layers.append(('LBISTA T=' + str(t + 1), xhat_, (lam_,gamma_)))

    """
    # check other functions in this file (e.g., build_LISTA and build_LAMP4SSC) to implement LBISTA network
    # send me questions if needed
    """

    return layers

def build_TiBLISTA(prob, T, initial_lambda=.1, initial_gamma=1, untied=False):
    """
    Builds a LISTA network to infer x from prob.y_ = matmul(prob.A,x) + AWGN
    prob            - is a TFGenerator which contains problem parameters and def of how to generate training data
    initial_lambda  - could be some parameter of Block ISTA <- DELETE if unnecessary
    untied          - flag for tied or untied case
    Return a list of layer info (name,xhat_,newvars)
     name : description, e.g. 'LISTA T=1'
     xhat_ : that which approximates x_ at some point in the algorithm
     newvars : a tuple of layer-specific trainable variables
    """
    blocksoft = block_soft_threshold
    layers = []
    A = prob.A
    M, N = A.shape
    #Wmat = sio.loadmat('W.mat') #W has to be precomputed before running the script
    W = A
    gamma = initial_gamma
    gamma0_ = tf.Variable(gamma, dtype=tf.float32, name='gamma_0')
    W_ = tf.Variable(np.transpose(W), dtype=tf.float32, trainable = True)
    #layers.append(('Linear', tf.matmul(gamma0_*W_, prob.y_), None))
    initial_lambda = np.array(initial_lambda).astype(np.float32)
    lam0_ = tf.Variable(initial_lambda, name='lam_0')
    xhat_ = blocksoft(tf.matmul(gamma0_*W_, prob.y_), lam0_)
    layers.append(('LBISTA T=1', xhat_, (lam0_,gamma0_,W_)))
    # pdb.set_trace()
    for t in range(1, T):
        lam_ = tf.Variable(initial_lambda, name='lam_{0}'.format(t))
        gamma_ = tf.Variable(gamma, dtype=tf.float32, name='gamma_{0}'.format(t))
        xhat_ = blocksoft(xhat_ - tf.matmul(gamma_*W_, tf.matmul(prob.A_, xhat_) - prob.y_), lam_)
        layers.append(('LBISTA T=' + str(t + 1), xhat_, (lam_,gamma_,W_)))

    """
    # check other functions in this file (e.g., build_LISTA and build_LAMP4SSC) to implement LBISTA network
    # send me questions if needed
    """

    return layers


def build_LBISTA_CPSS(prob, T, initial_lambda=.1, initial_gamma=1, untied=False):
    """
    Builds a LISTA network to infer x from prob.y_ = matmul(prob.A,x) + AWGN
    prob            - is a TFGenerator which contains problem parameters and def of how to generate training data
    initial_lambda  - could be some parameter of Block ISTA <- DELETE if unnecessary
    untied          - flag for tied or untied case
    Return a list of layer info (name,xhat_,newvars)
     name : description, e.g. 'LISTA T=1'
     xhat_ : that which approximates x_ at some point in the algorithm
     newvars : a tuple of layer-specific trainable variables
    """
    blocksoft = block_soft_threshold
    layers = []

    A = prob.A
    M, N = A.shape
    #Wmat = sio.loadmat('W.mat') #W has to be precomputed before running the script
    W = A
    gamma = initial_gamma
    gamma0_ = tf.Variable(gamma, dtype=tf.float32, name='gamma_0')
    W_ = tf.Variable(np.transpose(W), dtype=tf.float32, trainable = True)
    #layers.append(('Linear', tf.matmul(gamma0_*W_, prob.y_), None))
    initial_lambda = np.array(initial_lambda).astype(np.float32)
    lam0_ = tf.Variable(initial_lambda, name='lam_0')
    xhat_ = blocksoft(tf.matmul(gamma0_*W_, prob.y_), lam0_)
    layers.append(('LBISTA T=1', xhat_, (lam0_,gamma0_,W_)))
    # pdb.set_trace()
    for t in range(1, T):
        lam_ = tf.Variable(initial_lambda, name='lam_{0}'.format(t))
        W_ = tf.Variable(np.transpose(W), dtype=tf.float32, name='W_{0}'.format(t), trainable = True)
        gamma_ = tf.Variable(gamma, dtype=tf.float32, name='gamma_{0}'.format(t))
        xhat_ = blocksoft(xhat_ - tf.matmul(gamma_*W_, tf.matmul(prob.A_, xhat_) - prob.y_), lam_)
        layers.append(('LBISTA T=' + str(t + 1), xhat_, (lam_,gamma_,W_)))

    """
    # check other functions in this file (e.g., build_LISTA and build_LAMP4SSC) to implement LBISTA network
    # send me questions if needed
    """

    return layers

def build_BALISTA(prob, T, initial_lambda=.1, initial_gamma=1, untied=False):
    """
    Builds a LISTA network to infer x from prob.y_ = matmul(prob.A,x) + AWGN
    prob            - is a TFGenerator which contains problem parameters and def of how to generate training data
    initial_lambda  - could be some parameter of Block ISTA <- DELETE if unnecessary
    untied          - flag for tied or untied case
    Return a list of layer info (name,xhat_,newvars)
     name : description, e.g. 'LISTA T=1'
     xhat_ : that which approximates x_ at some point in the algorithm
     newvars : a tuple of layer-specific trainable variables
    """
    blocksoft = block_soft_threshold
    layers = []
    A = prob.A
    M, N = A.shape
    Wmat = sio.loadmat('Wblock.mat') #W has to be precomputed before running the script
    W = Wmat.get('W')
    gamma = initial_gamma
    gamma0_ = tf.Variable(gamma, dtype=tf.float32, name='gamma_0')
    W_ = tf.Variable(np.transpose(W), dtype=tf.float32, trainable = False)
    #layers.append(('Linear', tf.matmul(gamma0_*W_, prob.y_), None))
    initial_lambda = np.array(initial_lambda).astype(np.float32)
    lam0_ = tf.Variable(initial_lambda, name='lam_0')
    xhat_ = blocksoft(tf.matmul(gamma0_*W_, prob.y_), lam0_)
    layers.append(('LBISTA T=1', xhat_, (lam0_,gamma0_)))
    # pdb.set_trace()
    for t in range(1, T):
        lam_ = tf.Variable(initial_lambda, name='lam_{0}'.format(t))
        gamma_ = tf.Variable(gamma, dtype=tf.float32, name='gamma_{0}'.format(t))
        xhat_ = blocksoft(xhat_ - tf.matmul(gamma_*W_, tf.matmul(prob.A_, xhat_) - prob.y_), lam_)
        layers.append(('LBISTA T=' + str(t + 1), xhat_, (lam_,gamma_)))

    """
    # check other functions in this file (e.g., build_LISTA and build_LAMP4SSC) to implement LBISTA network
    # send me questions if needed
    """

    return layers, W


def build_BALISTA_v2(prob, T, initial_lambda=.1, initial_gamma=1, untied=False):
    """
    Builds a LISTA network to infer x from prob.y_ = matmul(prob.A,x) + AWGN
    prob            - is a TFGenerator which contains problem parameters and def of how to generate training data
    initial_lambda  - could be some parameter of Block ISTA <- DELETE if unnecessary
    untied          - flag for tied or untied case
    Return a list of layer info (name,xhat_,newvars)
     name : description, e.g. 'LISTA T=1'
     xhat_ : that which approximates x_ at some point in the algorithm
     newvars : a tuple of layer-specific trainable variables
    """
    blocksoft = block_soft_threshold
    layers = []
    A = prob.A
    gamma = initial_gamma
    gamma0_ = tf.Variable(gamma, dtype=tf.float32, name='gamma_0')

    W = bst.linalg.pinv(A.T)

    W_ = tf.Variable(W.T, dtype=tf.float32, trainable = False)
    #layers.append(('Linear', tf.matmul(gamma0_*W_, prob.y_), None))
    initial_lambda = np.array(initial_lambda).astype(np.float32)
    lam0_ = tf.Variable(initial_lambda, name='lam_0')
    xhat_ = blocksoft(tf.matmul(gamma0_*W_, prob.y_), lam0_)
    layers.append(('LBISTA T=1', xhat_, (lam0_, gamma0_)))
    # pdb.set_trace()
    for t in range(1, T):
        lam_ = tf.Variable(initial_lambda, name='lam_{0}'.format(t))
        gamma_ = tf.Variable(gamma, dtype=tf.float32, name='gamma_{0}'.format(t))
        xhat_ = blocksoft(xhat_ - tf.matmul(gamma_*W_, tf.matmul(prob.A_, xhat_) - prob.y_), lam_)
        layers.append(('LBISTA T=' + str(t + 1), xhat_, (lam_, gamma_)))

    """
    # check other functions in this file (e.g., build_LISTA and build_LAMP4SSC) to implement LBISTA network
    # send me questions if needed
    """

    return layers, W

def build_BALISTA_v3(prob, T, initial_lambda=.1, initial_gamma=1, untied=False):
    """
    Builds a LISTA network to infer x from prob.y_ = matmul(prob.A,x) + AWGN
    prob            - is a TFGenerator which contains problem parameters and def of how to generate training data
    initial_lambda  - could be some parameter of Block ISTA <- DELETE if unnecessary
    untied          - flag for tied or untied case
    Return a list of layer info (name,xhat_,newvars)
     name : description, e.g. 'LISTA T=1'
     xhat_ : that which approximates x_ at some point in the algorithm
     newvars : a tuple of layer-specific trainable variables
    """
    blocksoft = block_soft_threshold
    layers = []
    A = prob.A
    M, N = A.shape
    W = A
    gamma = initial_gamma
    gamma0_ = tf.Variable(gamma, dtype=tf.float32, name='gamma_0')

    W = bst.compute_W_v1(A, 32, 16)

    pdb.set_trace()

    W_ = tf.Variable(W.T, dtype=tf.float32, trainable = False)
    #layers.append(('Linear', tf.matmul(gamma0_*W_, prob.y_), None))
    initial_lambda = np.array(initial_lambda).astype(np.float32)
    lam0_ = tf.Variable(initial_lambda, name='lam_0')
    xhat_ = blocksoft(tf.matmul(gamma0_*W_, prob.y_), lam0_)
    layers.append(('LBISTA T=1', xhat_, (lam0_, gamma0_)))
    # pdb.set_trace()
    for t in range(1, T):
        lam_ = tf.Variable(initial_lambda, name='lam_{0}'.format(t))
        gamma_ = tf.Variable(gamma, dtype=tf.float32, name='gamma_{0}'.format(t))
        xhat_ = blocksoft(xhat_ - tf.matmul(gamma_*W_, tf.matmul(prob.A_, xhat_) - prob.y_), lam_)
        layers.append(('LBISTA T=' + str(t + 1), xhat_, (lam_,gamma_)))

    """
    # check other functions in this file (e.g., build_LISTA and build_LAMP4SSC) to implement LBISTA network
    # send me questions if needed
    """

    return layers, W

def build_BALISTA_v4(prob, T, initial_lambda=.1, initial_gamma=1, untied=False):
    """
    Builds a LISTA network to infer x from prob.y_ = matmul(prob.A,x) + AWGN
    prob            - is a TFGenerator which contains problem parameters and def of how to generate training data
    initial_lambda  - could be some parameter of Block ISTA <- DELETE if unnecessary
    untied          - flag for tied or untied case
    Return a list of layer info (name,xhat_,newvars)
     name : description, e.g. 'LISTA T=1'
     xhat_ : that which approximates x_ at some point in the algorithm
     newvars : a tuple of layer-specific trainable variables
    """
    blocksoft = block_soft_threshold
    layers = []
    A = prob.A
    gamma = initial_gamma
    gamma0_ = tf.Variable(gamma, dtype=tf.float32, name='gamma_0')
    # pdb.set_trace()
    #W = bst.compute_W_Lagrange(A, 32, 3)

    #solve the following problem to get the analytical weight matrix
    # D = prob.A
    # n = prob.L
    # d = prob.B
    # m = prob.m
    # B_up = cp.Variable((m, n * d))
    #
    # I = np.kron(np.eye(n), np.ones((d, d)))
    # k = np.tile(np.eye(d), (1, n)).T
    # b = cp.multiply(B_up.T @ D, I)
    # b = cp.matmul(b, k)  # extracting the diagonal blocks of D^TB
    # constraints = [b == k]
    # objective = cp.Minimize(1 / d * (cp.norm(B_up.T @ D, 'fro')))
    # prob = cp.Problem(objective, constraints)
    # result = prob.solve()
    #
    # W = B_up.value
    Wmat = sio.loadmat('sparse_case_W_up.mat')
    W = Wmat.get('W')
    W_ = tf.Variable(W.T, dtype=tf.float32, trainable = False)
    #layers.append(('Linear', tf.matmul(gamma0_*W_, prob.y_), None))
    initial_lambda = np.array(initial_lambda).astype(np.float32)
    lam0_ = tf.Variable(initial_lambda, name='lam_0')
    xhat_ = blocksoft(tf.matmul(tf.math.abs(gamma0_)*W_, prob.y_), lam0_)
    layers.append(('LBISTA T=1', xhat_, (lam0_, gamma0_)))
    # pdb.set_trace()
    for t in range(1, T):
        lam_ = tf.Variable(initial_lambda, name='lam_{0}'.format(t))
        gamma_ = tf.Variable(gamma, dtype=tf.float32, name='gamma_{0}'.format(t))
        xhat_ = blocksoft(xhat_ - tf.matmul(tf.math.abs(gamma_)*W_, tf.matmul(prob.A_, xhat_) - prob.y_), lam_)
        layers.append(('LBISTA T=' + str(t + 1), xhat_, (lam_,gamma_)))

    """
    # check other functions in this file (e.g., build_LISTA and build_LAMP4SSC) to implement LBISTA network
    # send me questions if needed
    """

    return layers, W

def build_BALISTA_v5(prob, T, initial_lambda=.1, initial_gamma=1, untied=False):
    """
    Builds a LISTA network to infer x from prob.y_ = matmul(prob.A,x) + AWGN
    prob            - is a TFGenerator which contains problem parameters and def of how to generate training data
    initial_lambda  - could be some parameter of Block ISTA <- DELETE if unnecessary
    untied          - flag for tied or untied case
    Return a list of layer info (name,xhat_,newvars)
     name : description, e.g. 'LISTA T=1'
     xhat_ : that which approximates x_ at some point in the algorithm
     newvars : a tuple of layer-specific trainable variables
    """
    blocksoft = block_soft_threshold
    layers = []
    A = prob.A
    gamma = initial_gamma
    gamma0_ = tf.Variable(gamma, dtype=tf.float32, name='gamma_0')
    # pdb.set_trace()
    #W = bst.compute_W_Lagrange(A, 32, 3)
    # solve the following problem to get the analytical weight matrix
    # D = prob.A
    # n = prob.L
    # d = prob.B
    # m = prob.m
    # B_cvx = cp.Variable((m, n * d))
    #
    # I = np.kron(np.eye(n), np.ones((d, d)))
    # k = np.tile(np.eye(d), (1, n)).T
    # b = cp.multiply(B_cvx.T @ D, I)
    # b = cp.matmul(b, k)  # extracting the diagonal blocks of D^TB
    # constraints = [b == k]
    # objective = cp.Minimize(1/d*cp.max(R(B_cvx.T@D-np.eye(n*d), n, d)))
    # prob = cp.Problem(objective, constraints)
    # result = prob.solve()
    # W = B_cvx.value
    Wmat = sio.loadmat('sparse_case_W_cvx.mat')
    W = Wmat.get('W_cvx')
    W_ = tf.Variable(W.T, dtype=tf.float32, trainable = False)
    #layers.append(('Linear', tf.matmul(gamma0_*W_, prob.y_), None))
    initial_lambda = np.array(initial_lambda).astype(np.float32)
    lam0_ = tf.Variable(initial_lambda, name='lam_0')
    xhat_ = blocksoft(tf.matmul(tf.math.abs(gamma0_)*W_, prob.y_), lam0_)
    layers.append(('LBISTA T=1', xhat_, (lam0_, gamma0_)))
    # pdb.set_trace()
    for t in range(1, T):
        lam_ = tf.Variable(initial_lambda, name='lam_{0}'.format(t))
        gamma_ = tf.Variable(gamma, dtype=tf.float32, name='gamma_{0}'.format(t))
        xhat_ = blocksoft(xhat_ - tf.matmul(tf.math.abs(gamma_)*W_, tf.matmul(prob.A_, xhat_) - prob.y_), lam_)
        layers.append(('LBISTA T=' + str(t + 1), xhat_, (lam_,gamma_)))

    """
    # check other functions in this file (e.g., build_LISTA and build_LAMP4SSC) to implement LBISTA network
    # send me questions if needed
    """

    return layers, W

def build_TiLISTA(prob, T, initial_lambda=.1, initial_gamma=1, untied=False):
    """
    Builds a LISTA network to infer x from prob.y_ = matmul(prob.A,x) + AWGN
    prob            - is a TFGenerator which contains problem parameters and def of how to generate training data
    initial_lambda  - could be some parameter of Block ISTA <- DELETE if unnecessary
    untied          - flag for tied or untied case
    Return a list of layer info (name,xhat_,newvars)
     name : description, e.g. 'LISTA T=1'
     xhat_ : that which approximates x_ at some point in the algorithm
     newvars : a tuple of layer-specific trainable variables
    """
    blocksoft = simple_soft_threshold
    layers = []
    A = prob.A
    M, N = A.shape
    #Wmat = sio.loadmat('W.mat') #W has to be precomputed before running the script
    W = A
    gamma = initial_gamma
    gamma0_ = tf.Variable(gamma, dtype=tf.float32, name='gamma_0')
    W_ = tf.Variable(np.transpose(W), dtype=tf.float32, trainable = True)
    layers.append(('Linear', tf.matmul(2*gamma0_*W_, prob.y_), None))
    initial_lambda = np.array(initial_lambda).astype(np.float32)
    lam0_ = tf.Variable(initial_lambda, name='lam_0')
    xhat_ = blocksoft(tf.matmul(2*gamma0_*W_, prob.y_), lam0_)
    layers.append(('LBISTA T=1', xhat_, (lam0_,gamma0_)))
    # pdb.set_trace()
    for t in range(1, T):
        lam_ = tf.Variable(initial_lambda, name='lam_{0}'.format(t))
        gamma_ = tf.Variable(gamma, dtype=tf.float32, name='gamma_{0}'.format(t))
        xhat_ = blocksoft(xhat_ - tf.matmul(2*gamma_*W_, tf.matmul(prob.A_, xhat_) - prob.y_), lam_)
        layers.append(('LBISTA T=' + str(t + 1), xhat_, (lam_,gamma_)))

    """
    # check other functions in this file (e.g., build_LISTA and build_LAMP4SSC) to implement LBISTA network
    # send me questions if needed
    """

    return layers


def build_TiBLISTA(prob, T, initial_lambda=.1, initial_gamma=1, untied=False):
    """
    Builds a LISTA network to infer x from prob.y_ = matmul(prob.A,x) + AWGN
    prob            - is a TFGenerator which contains problem parameters and def of how to generate training data
    initial_lambda  - could be some parameter of Block ISTA <- DELETE if unnecessary
    untied          - flag for tied or untied case
    Return a list of layer info (name,xhat_,newvars)
     name : description, e.g. 'LISTA T=1'
     xhat_ : that which approximates x_ at some point in the algorithm
     newvars : a tuple of layer-specific trainable variables
    """
    blocksoft = block_soft_threshold
    layers = []
    A = prob.A
    M, N = A.shape
    #Wmat = sio.loadmat('W.mat') #W has to be precomputed before running the script
    W = A
    gamma = initial_gamma
    gamma0_ = tf.Variable(gamma, dtype=tf.float32, name='gamma_0')
    W_ = tf.Variable(np.transpose(W), dtype=tf.float32, trainable = True)
    layers.append(('Linear', tf.matmul(gamma0_*W_, prob.y_), None))
    initial_lambda = np.array(initial_lambda).astype(np.float32)
    lam0_ = tf.Variable(initial_lambda, name='lam_0')
    xhat_ = blocksoft(tf.matmul(gamma0_*W_, prob.y_), lam0_)
    layers.append(('LBISTA T=1', xhat_, (lam0_,gamma0_)))
    # pdb.set_trace()
    for t in range(1, T):
        lam_ = tf.Variable(initial_lambda, name='lam_{0}'.format(t))
        gamma_ = tf.Variable(gamma, dtype=tf.float32, name='gamma_{0}'.format(t))
        xhat_ = blocksoft(xhat_ - tf.matmul(gamma_*W_, tf.matmul(prob.A_, xhat_) - prob.y_), lam_)
        layers.append(('LBISTA T=' + str(t + 1), xhat_, (lam_,gamma_)))

    """
    # check other functions in this file (e.g., build_LISTA and build_LAMP4SSC) to implement LBISTA network
    # send me questions if needed
    """

    return layers

def build_LBFISTA(prob,T,initial_lambda=.1,untied=False):
    """
    Builds a LISTA network to infer x from prob.y_ = matmul(prob.A,x) + AWGN
    prob            - is a TFGenerator which contains problem parameters and def of how to generate training data
    initial_lambda  - could be some parameter of Block ISTA <- DELETE if unnecessary
    untied          - flag for tied or untied case
    Return a list of layer info (name,xhat_,newvars)
     name : description, e.g. 'LISTA T=1'
     xhat_ : that which approximates x_ at some point in the algorithm
     newvars : a tuple of layer-specific trainable variables
    """
    blocksoft=block_soft_threshold
    layers = []
    A = prob.A
    M,N = A.shape
    B = A.T / (1.01 * la.norm(A,2)**2)
    B_ =  tf.Variable(B,dtype=tf.float32,name='B_0')
    By_ = tf.matmul(B_,prob.y_)
    S_ = tf.Variable( np.identity(N) - np.matmul(B,A),dtype=tf.float32,name='S_0')
    #pdb.set_trace()
    layers.append( ('Linear',By_,None) )
    
    initial_lambda = np.array(initial_lambda).astype(np.float32)
    #if getattr(prob,'iid',True) == False:
    #    # create a parameter for each coordinate in x
    #    initial_lambda = initial_lambda*np.ones( (N,1),dtype=np.float32 )
    lam0_ = tf.Variable( initial_lambda,name='lam_0')
    xhat_ = blocksoft( By_, lam0_)
    tk = (1+np.sqrt(1+4*1**2))*2**(-1)
    z_ = xhat_
    layers.append( ('LBISTA T=1',xhat_, (lam0_,) ) )
    #pdb.set_trace()
    for t in range(1,T):
        t_prev = tk
        xhat_prev_ = xhat_ 
        lam_ = tf.Variable( initial_lambda,name='lam_{0}'.format(t) )
        xhat_ = blocksoft( tf.matmul(S_,z_) + By_, lam_ )
        tk = (1+np.sqrt(1+4*t_prev**2))*2**(-1)
        z_ = xhat_ + (t_prev-1)*(tk)**(-1)*(xhat_-xhat_prev_)
        layers.append( ('LBISTA T='+str(t+1),xhat_,(lam_,)) )

    """
    # check other functions in this file (e.g., build_LISTA and build_LAMP4SSC) to implement LBISTA network
    # send me questions if needed
    """

    return layers


def build_LBFISTA_idea(prob, T, initial_lambda=.1, untied=False):
    """
    Builds a LISTA network to infer x from prob.y_ = matmul(prob.A,x) + AWGN
    prob            - is a TFGenerator which contains problem parameters and def of how to generate training data
    initial_lambda  - could be some parameter of Block ISTA <- DELETE if unnecessary
    untied          - flag for tied or untied case
    Return a list of layer info (name,xhat_,newvars)
     name : description, e.g. 'LISTA T=1'
     xhat_ : that which approximates x_ at some point in the algorithm
     newvars : a tuple of layer-specific trainable variables
    """
    blocksoft = block_soft_threshold
    layers = []
    A = prob.A
    M, N = A.shape
    B = A.T / (1.01 * la.norm(A, 2) ** 2)
    B_ = tf.Variable(B, dtype=tf.float32, name='B_0')
    By_ = tf.matmul(B_, prob.y_)
    S_ = tf.Variable(np.identity(N) - np.matmul(B, A), dtype=tf.float32, name='S_0')
    # pdb.set_trace()
    layers.append(('Linear', By_, None))

    initial_lambda = np.array(initial_lambda).astype(np.float32)
    # if getattr(prob,'iid',True) == False:
    #    # create a parameter for each coordinate in x
    #    initial_lambda = initial_lambda*np.ones( (N,1),dtype=np.float32 )
    lam0_ = tf.Variable(initial_lambda, name='lam_0')
    xhat_ = blocksoft(By_, lam0_)
    tk_ = tf.Variable((1 + np.sqrt(1 + 4 * 1 ** 2)) * 2 ** (-1), name='t_0')
    z_ = xhat_
    layers.append(('LBISTA T=1', xhat_, (lam0_,tk_)))
    # pdb.set_trace()
    for t in range(1, T):
        t_prev = tk_.read_value()
        t_prev_ = tk_
        xhat_prev_ = xhat_
        lam_ = tf.Variable(initial_lambda, name='lam_{0}'.format(t))
        xhat_ = blocksoft(tf.matmul(S_, z_) + By_, lam_)
        tk_ = tf.Variable((1 + tf.sqrt(1 + 4 * t_prev_ ** 2)) * 2 ** (-1), name='t_{0}'.format(t))
        z_ = xhat_ + (t_prev_ - 1) * (tk_) ** (-1) * (xhat_ - xhat_prev_)
        layers.append(('LBISTA T=' + str(t + 1), xhat_, (lam_,tk_)))

    """
    # check other functions in this file (e.g., build_LISTA and build_LAMP4SSC) to implement LBISTA network
    # send me questions if needed
    """

    return layers

def build_LBelastic_net(prob,T,initial_lambda=.1,untied=False):
    """
    Builds a LISTA network to infer x from prob.y_ = matmul(prob.A,x) + AWGN
    prob            - is a TFGenerator which contains problem parameters and def of how to generate training data
    initial_lambda  - could be some parameter of Block ISTA <- DELETE if unnecessary
    untied          - flag for tied or untied case
    Return a list of layer info (name,xhat_,newvars)
     name : description, e.g. 'LISTA T=1'
     xhat_ : that which approximates x_ at some point in the algorithm
     newvars : a tuple of layer-specific trainable variables
    """
    eta=block_soft_threshold_elastic_net
    layers = []
    A = prob.A
    M,N = A.shape
    B = A.T / (1.01 * la.norm(A,2)**2)
    B_ =  tf.Variable(B,dtype=tf.float32,name='B_0')
    By_ = tf.matmul(B_,prob.y_)
    S_ = tf.Variable( np.identity(N) - np.matmul(B,A),dtype=tf.float32,name='S_0')
    #pdb.set_trace()
    layers.append( ('Linear',By_,None) )
    
    initial_lambda = np.array(initial_lambda).astype(np.float32)
    #if getattr(prob,'iid',True) == False:
    #    # create a parameter for each coordinate in x
    #    initial_lambda = initial_lambda*np.ones( (N,1),dtype=np.float32 )
    al0_ = tf.Variable( initial_lambda,name='al_0')
    lam0_ = tf.Variable( initial_lambda,name='lam_0')
    xhat_ = eta( By_, al0_, lam0_)
    layers.append( ('LBISTA T=1',xhat_, (al0_, lam0_) ) )
    #pdb.set_trace()
    for t in range(1,T):
        al_ = tf.Variable( initial_lambda,name='al_{0}'.format(t) )
        lam_ = tf.Variable( initial_lambda,name='lam_{0}'.format(t) )
        xhat_ = eta( tf.matmul(S_,xhat_) + By_, al_, lam_ )
        layers.append( ('LBISTA T='+str(t+1),xhat_,(al_, lam_)) )

    """
    # check other functions in this file (e.g., build_LISTA and build_LAMP4SSC) to implement LBISTA network
    # send me questions if needed
    """

    return layers
  
def build_UntiedLBelastic_net(prob,T,initial_lambda=.1,untied=False):
    """
    Builds a LISTA network to infer x from prob.y_ = matmul(prob.A,x) + AWGN
    prob            - is a TFGenerator which contains problem parameters and def of how to generate training data
    initial_lambda  - could be some parameter of Block ISTA <- DELETE if unnecessary
    untied          - flag for tied or untied case
    Return a list of layer info (name,xhat_,newvars)
     name : description, e.g. 'LISTA T=1'
     xhat_ : that which approximates x_ at some point in the algorithm
     newvars : a tuple of layer-specific trainable variables
    """
    eta=block_soft_threshold_elastic_net
    layers = []
    A = prob.A
    M,N = A.shape
    B = A.T / (1.01 * la.norm(A,2)**2)
    B_ =  tf.Variable(B,dtype=tf.float32,name='B_0')
    By_ = tf.matmul(B_,prob.y_)
    S_ = tf.Variable( np.identity(N) - np.matmul(B,A),dtype=tf.float32,name='S_0')
    #pdb.set_trace()
    #layers.append( ('Linear',By_,None) )
    
    initial_lambda = np.array(initial_lambda).astype(np.float32)
    #if getattr(prob,'iid',True) == False:
    #    # create a parameter for each coordinate in x
    #    initial_lambda = initial_lambda*np.ones( (N,1),dtype=np.float32 )
    al0_ = tf.Variable( initial_lambda,name='al_0')
    lam0_ = tf.Variable( initial_lambda,name='lam_0')
    xhat_ = eta( By_, al0_, lam0_)
    layers.append( ('LBISTA T=1',xhat_, (al0_, lam0_, B_, S_) ) )
    #pdb.set_trace()
    for t in range(1,T):
        al_ = tf.Variable( initial_lambda,name='al_{0}'.format(t) )
        lam_ = tf.Variable( initial_lambda,name='lam_{0}'.format(t) )
        B_ =  tf.Variable(B,dtype=tf.float32,name='B_{0}'.format(t) )
        By_ = tf.matmul(B_,prob.y_)
        S_ = tf.Variable( np.identity(N) - np.matmul(B,A),dtype=tf.float32,name='S_{0}'.format(t) )
        xhat_ = eta( tf.matmul(S_,xhat_) + By_, al_, lam_ )
        layers.append( ('LBISTA T='+str(t+1),xhat_,(al_, lam_, B_, S_)) )

    """
    # check other functions in this file (e.g., build_LISTA and build_LAMP4SSC) to implement LBISTA network
    # send me questions if needed
    """

    return layers

def build_LBFastelastic_net(prob,T,initial_lambda=.1,untied=False):
    """
    Builds a LISTA network to infer x from prob.y_ = matmul(prob.A,x) + AWGN
    prob            - is a TFGenerator which contains problem parameters and def of how to generate training data
    initial_lambda  - could be some parameter of Block ISTA <- DELETE if unnecessary
    untied          - flag for tied or untied case
    Return a list of layer info (name,xhat_,newvars)
     name : description, e.g. 'LISTA T=1'
     xhat_ : that which approximates x_ at some point in the algorithm
     newvars : a tuple of layer-specific trainable variables
    """
    eta=block_soft_threshold_elastic_net
    layers = []
    A = prob.A
    M,N = A.shape
    B = A.T / (1.01 * la.norm(A,2)**2)
    B_ =  tf.Variable(B,dtype=tf.float32,name='B_0')
    By_ = tf.matmul(B_,prob.y_)
    S_ = tf.Variable( np.identity(N) - np.matmul(B,A),dtype=tf.float32,name='S_0')
    #pdb.set_trace()
    layers.append( ('Linear',By_,None) )
    
    initial_lambda = np.array(initial_lambda).astype(np.float32)
    #if getattr(prob,'iid',True) == False:
    #    # create a parameter for each coordinate in x
    #    initial_lambda = initial_lambda*np.ones( (N,1),dtype=np.float32 )
    al0_ = tf.Variable( initial_lambda,name='al_0')
    lam0_ = tf.Variable( initial_lambda,name='lam_0')
    xhat_ = eta( By_, al0_, lam0_)
    tk = (1+np.sqrt(1+4*1**2))*2**(-1)
    z_ = xhat_
    layers.append( ('LBISTA T=1',xhat_, (al0_, lam0_) ) )
    #pdb.set_trace()
    for t in range(1,T):
        t_prev = tk
        xhat_prev_ = xhat_ 
        al_ = tf.Variable( initial_lambda,name='al_{0}'.format(t) )
        lam_ = tf.Variable( initial_lambda,name='lam_{0}'.format(t) )
        xhat_ = eta( tf.matmul(S_,z_) + By_, al_, lam_ )
        tk = (1+np.sqrt(1+4*t_prev**2))*2**(-1)
        z_ = xhat_ + (t_prev-1)*(tk)**(-1)*(xhat_-xhat_prev_)
        layers.append( ('LBISTA T='+str(t+1),xhat_,(al_, lam_)) )

    """
    # check other functions in this file (e.g., build_LISTA and build_LAMP4SSC) to implement LBISTA network
    # send me questions if needed
    """

    return layers


def build_LISTA(prob,T,initial_lambda=.1,untied=False):
    """
    Builds a LISTA network to infer x from prob.y_ = matmul(prob.A,x) + AWGN
    return a list of layer info (name,xhat_,newvars)
     name : description, e.g. 'LISTA T=1'
     xhat_ : that which approximates x_ at some point in the algorithm
     newvars : a tuple of layer-specific trainable variables
    """
    assert not untied,'TODO: untied'
    eta = simple_soft_threshold
    layers = []
    A = prob.A
    M,N = A.shape
    B = A.T / (1.01 * la.norm(A,2)**2)
    B_ =  tf.Variable(B,dtype=tf.float32,name='B_0')
    S_ = tf.Variable( np.identity(N) - np.matmul(B,A),dtype=tf.float32,name='S_0')
    By_ = tf.matmul( B_ , prob.y_ )
    layers.append( ('Linear',By_,None) )

    initial_lambda = np.array(initial_lambda).astype(np.float32)
    if getattr(prob,'iid',True) == False:
        # create a parameter for each coordinate in x
        initial_lambda = initial_lambda*np.ones( (N,1),dtype=np.float32 )
    lam0_ = tf.Variable( initial_lambda,name='lam_0')
    xhat_ = eta( By_, lam0_)
    #pdb.set_trace()
    layers.append( ('LISTA T=1',xhat_, (lam0_,) ) )
    for t in range(1,T):
        lam_ = tf.Variable( initial_lambda,name='lam_{0}'.format(t) )
        xhat_ = eta( tf.matmul(S_,xhat_) + By_, lam_ )
        layers.append( ('LISTA T='+str(t+1),xhat_,(lam_,)) )
    return layers
