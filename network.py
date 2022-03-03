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

def block_soft_threshold(X_, al_, prob):
    L = prob.L
    B = prob.B
    shape = K.shape(X_)
    pool_shape1 = tf.stack([L, B, shape[1]])
    al_ = tf.maximum(al_, 0)
    Xnew_ = K.reshape(X_, pool_shape1)  # i'th coloumn of X_
    r_ = tf.sqrt(tf.reduce_sum((Xnew_ ** 2), 1))
    r_ = tf.maximum(.0, 1 - tf.math.divide_no_nan(al_ , r_))
    shaper = K.shape(r_)
    pool_r = tf.stack([L, 1, shaper[1]])
    r_ = K.reshape(r_, pool_r)  # Diesen reshape Befehl auch zu K.reshape ändern
    pool_shape2 = tf.stack([B * L, shape[1]])

    Xnew_ = tf.multiply(Xnew_, r_)

    return K.reshape(Xnew_, pool_shape2)  # transpose, bc of the list and tf.stack() operator

def block_soft_threshold_elastic_net(X_, al_, la_, prob):
    L = prob.L
    B = prob.B

    shape = K.shape(X_)
    pool_shape1 = tf.stack([L,B, shape[1]])
    al_ = tf.maximum(al_, 0)
    Xnew_ = K.reshape(X_, pool_shape1)  # i'th coloumn of X_
    r_ = tf.sqrt(tf.reduce_sum((Xnew_ ** 2), 1))
    r_ = tf.maximum(.0, 1 - tf.math.divide_no_nan(al_ , r_))
    shaper = K.shape(r_)
    pool_r = tf.stack([L, 1, shaper[1]])
    r_ = K.reshape(r_, pool_r)  # Diesen reshape Befehl auch zu K.reshape ändern
    pool_shape2 = tf.stack([B * L, shape[1]])

    Xnew_ = tf.multiply(Xnew_, r_)*(1+la_)**(-1)

    return K.reshape(Xnew_, pool_shape2)  # transpose, bc of the list and tf.stack() operator

def build_LBISTA(prob,T,initial_lambda=.1):
    """
    Builds a LBISTA network to infer x from prob.y_ = matmul(prob.A,x) + AWGN
    prob            - is a TFGenerator which contains problem parameters and def of how to generate training data
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

    initial_lambda = np.array(initial_lambda).astype(np.float32)

    lam0_ = tf.Variable( initial_lambda,name='lam_0')
    xhat_ = blocksoft( By_, lam0_, prob)
    layers.append( ('LBISTA T=1',xhat_, (lam0_,B_, S_) ) )
    for t in range(1,T):
        lam_ = tf.Variable( initial_lambda,name='lam_{0}'.format(t) )
        xhat_ = blocksoft( tf.matmul(S_,xhat_) + tf.matmul(B_,prob.y_), lam_ , prob)
        layers.append( ('LBISTA T='+str(t+1),xhat_,(lam_,B_, S_)) )

    return layers


def build_LBISTA_untied(prob,T,initial_lambda=.1):
    """
    Builds a LBISTA network to infer x from prob.y_ = matmul(prob.A,x) + AWGN
    prob            - is a TFGenerator which contains problem parameters and def of how to generate training data
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

    initial_lambda = np.array(initial_lambda).astype(np.float32)

    lam0_ = tf.Variable( initial_lambda,name='lam_0')
    xhat_ = blocksoft( By_, lam0_, prob)
    layers.append( ('LBISTA T=1',xhat_, (lam0_,B_, S_) ) )
    #pdb.set_trace()
    for t in range(1,T):
        B_ =  tf.Variable(B,dtype=tf.float32,name='B_{0}'.format(t) )
        S_ = tf.Variable( np.identity(N) - np.matmul(B,A),dtype=tf.float32,name='S_{0}'.format(t) )
        lam_ = tf.Variable( initial_lambda,name='lam_{0}'.format(t) )
        xhat_ = blocksoft( tf.matmul(S_,xhat_) + tf.matmul(B_,prob.y_), lam_ , prob)
        layers.append( ('LBISTA T='+str(t+1),xhat_,(lam_,B_, S_)) )

    return layers

def build_LBISTA_CPSS(prob, T, initial_lambda=.1, initial_gamma=1, untied=False):
    """
    Builds a LBISTA_CPSS network to infer x from prob.y_ = matmul(prob.A,x) + AWGN
    prob            - is a TFGenerator which contains problem parameters and def of how to generate training data
    untied          - flag for tied or untied case
    Return a list of layer info (name,xhat_,newvars)
     name : description, e.g. 'LISTA T=1'
     xhat_ : that which approximates x_ at some point in the algorithm
     newvars : a tuple of layer-specific trainable variables
    """
    blocksoft = block_soft_threshold
    layers = []

    A = prob.A

    W = A
    gamma = initial_gamma
    gamma0_ = tf.Variable(gamma, dtype=tf.float32, name='gamma_0')
    W_ = tf.Variable(np.transpose(W), dtype=tf.float32, trainable = True)
    initial_lambda = np.array(initial_lambda).astype(np.float32)
    lam0_ = tf.Variable(initial_lambda, name='lam_0')
    xhat_ = blocksoft(tf.matmul(gamma0_*W_, prob.y_), lam0_, prob)
    layers.append(('LBISTA_CPSS T=1', xhat_, (lam0_,gamma0_,W_)))
    # pdb.set_trace()
    for t in range(1, T):
        lam_ = tf.Variable(initial_lambda, name='lam_{0}'.format(t))
        W_ = tf.Variable(np.transpose(W), dtype=tf.float32, name='W_{0}'.format(t), trainable = True)
        gamma_ = tf.Variable(gamma, dtype=tf.float32, name='gamma_{0}'.format(t))
        xhat_ = blocksoft(xhat_ - tf.matmul(gamma_*W_, tf.matmul(prob.A_, xhat_) - prob.y_), lam_, prob)
        layers.append(('LBISTA_CPSS T=' + str(t + 1), xhat_, (lam_,gamma_,W_)))

    return layers

def build_BALISTA_v4(prob, T, initial_lambda=.1, initial_gamma=1, untied=False):
    """
    Builds a BALISTA network to infer x from prob.y_ = matmul(prob.A,x) + AWGN
    prob            - is a TFGenerator which contains problem parameters and def of how to generate training data
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

    Wmat = sio.loadmat('small_normalized_W_up.mat')
    W = Wmat.get('W')
    W_ = tf.Variable(W.T, dtype=tf.float32, trainable = False)
    #layers.append(('Linear', tf.matmul(gamma0_*W_, prob.y_), None))
    initial_lambda = np.array(initial_lambda).astype(np.float32)
    lam0_ = tf.Variable(initial_lambda, name='lam_0')
    xhat_ = blocksoft(tf.matmul(tf.math.abs(gamma0_)*W_, prob.y_), lam0_, prob)
    layers.append(('BALISTA T=1', xhat_, (lam0_, gamma0_)))
    # pdb.set_trace()
    for t in range(1, T):
        lam_ = tf.Variable(initial_lambda, name='lam_{0}'.format(t))
        gamma_ = tf.Variable(gamma, dtype=tf.float32, name='gamma_{0}'.format(t))
        xhat_ = blocksoft(xhat_ - tf.matmul(tf.math.abs(gamma_)*W_, tf.matmul(prob.A_, xhat_) - prob.y_), lam_, prob)
        layers.append(('BALISTA T=' + str(t + 1), xhat_, (lam_,gamma_)))

    return layers, W

def build_BALISTA_v5(prob, T, initial_lambda=.1, initial_gamma=1, untied=False):
    """
    Builds a BALISTA network to infer x from prob.y_ = matmul(prob.A,x) + AWGN
    prob            - is a TFGenerator which contains problem parameters and def of how to generate training data
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

    Wmat = sio.loadmat('small_normalized_W_cvx.mat')
    W = Wmat.get('W_cvx')
    W_ = tf.Variable(W.T, dtype=tf.float32, trainable = False)

    initial_lambda = np.array(initial_lambda).astype(np.float32)
    lam0_ = tf.Variable(initial_lambda, name='lam_0')
    xhat_ = blocksoft(tf.matmul(tf.math.abs(gamma0_)*W_, prob.y_), lam0_, prob)
    layers.append(('BALISTA T=1', xhat_, (lam0_, gamma0_)))
    # pdb.set_trace()
    for t in range(1, T):
        lam_ = tf.Variable(initial_lambda, name='lam_{0}'.format(t))
        gamma_ = tf.Variable(gamma, dtype=tf.float32, name='gamma_{0}'.format(t))
        xhat_ = blocksoft(xhat_ - tf.matmul(tf.math.abs(gamma_)*W_, tf.matmul(prob.A_, xhat_) - prob.y_), lam_, prob)
        layers.append(('BALISTA T=' + str(t + 1), xhat_, (lam_,gamma_)))

    return layers, W

def build_TiBLISTA(prob, T, initial_lambda=.1, initial_gamma=1, untied=False):
    """
    Builds a TiBLISTA network to infer x from prob.y_ = matmul(prob.A,x) + AWGN
    prob            - is a TFGenerator which contains problem parameters and def of how to generate training data
    untied          - flag for tied or untied case
    Return a list of layer info (name,xhat_,newvars)
     name : description, e.g. 'LISTA T=1'
     xhat_ : that which approximates x_ at some point in the algorithm
     newvars : a tuple of layer-specific trainable variables
    """
    blocksoft = block_soft_threshold
    layers = []
    A = prob.A
    W = A
    gamma = initial_gamma
    gamma0_ = tf.Variable(gamma, dtype=tf.float32, name='gamma_0')
    W_ = tf.Variable(np.transpose(W), dtype=tf.float32, trainable = True)
    layers.append(('Linear', tf.matmul(gamma0_*W_, prob.y_), None))
    initial_lambda = np.array(initial_lambda).astype(np.float32)
    lam0_ = tf.Variable(initial_lambda, name='lam_0')
    xhat_ = blocksoft(tf.matmul(gamma0_*W_, prob.y_), lam0_, prob)
    layers.append(('TiBLISTA T=1', xhat_, (lam0_,gamma0_)))
    # pdb.set_trace()
    for t in range(1, T):
        lam_ = tf.Variable(initial_lambda, name='lam_{0}'.format(t))
        gamma_ = tf.Variable(gamma, dtype=tf.float32, name='gamma_{0}'.format(t))
        xhat_ = blocksoft(xhat_ - tf.matmul(gamma_*W_, tf.matmul(prob.A_, xhat_) - prob.y_), lam_, prob)
        layers.append(('TiBLISTA T=' + str(t + 1), xhat_, (lam_,gamma_)))

    return layers

def build_LBFISTA(prob,T,initial_lambda=.1):
    """
    Builds a LBFISTA network to infer x from prob.y_ = matmul(prob.A,x) + AWGN
    prob            - is a TFGenerator which contains problem parameters and def of how to generate training data
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

    lam0_ = tf.Variable( initial_lambda,name='lam_0')
    xhat_ = blocksoft( By_, lam0_, prob)
    tk = (1+np.sqrt(1+4*1**2))*2**(-1)
    z_ = xhat_
    layers.append( ('LBFISTA T=1',xhat_, (lam0_,) ) )
    #pdb.set_trace()
    for t in range(1,T):
        t_prev = tk
        xhat_prev_ = xhat_ 
        lam_ = tf.Variable( initial_lambda,name='lam_{0}'.format(t) )
        xhat_ = blocksoft( tf.matmul(S_,z_) + By_, lam_, prob)
        tk = (1+np.sqrt(1+4*t_prev**2))*2**(-1)
        z_ = xhat_ + (t_prev-1)*(tk)**(-1)*(xhat_-xhat_prev_)
        layers.append( ('LBFISTA T='+str(t+1),xhat_,(lam_,)) )

    return layers

def build_LBelastic_net(prob,T,initial_lambda=.1):
    """
    Builds a LBelastic_net network to infer x from prob.y_ = matmul(prob.A,x) + AWGN
    prob            - is a TFGenerator which contains problem parameters and def of how to generate training data
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

    al0_ = tf.Variable( initial_lambda,name='al_0')
    lam0_ = tf.Variable( initial_lambda,name='lam_0')
    xhat_ = eta( By_, al0_, lam0_, prob)
    layers.append( ('LBelastic_net T=1',xhat_, (al0_, lam0_) ) )
    #pdb.set_trace()
    for t in range(1,T):
        al_ = tf.Variable( initial_lambda,name='al_{0}'.format(t) )
        lam_ = tf.Variable( initial_lambda,name='lam_{0}'.format(t) )
        xhat_ = eta( tf.matmul(S_,xhat_) + By_, al_, lam_, prob)
        layers.append( ('LBelastic_net T='+str(t+1),xhat_,(al_, lam_)) )

    return layers
  
def build_UntiedLBelastic_net(prob,T,initial_lambda=.1):
    """
    Builds a LBelastic_net network to infer x from prob.y_ = matmul(prob.A,x) + AWGN
    prob            - is a TFGenerator which contains problem parameters and def of how to generate training data
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

    
    initial_lambda = np.array(initial_lambda).astype(np.float32)

    al0_ = tf.Variable( initial_lambda,name='al_0')
    lam0_ = tf.Variable( initial_lambda,name='lam_0')
    xhat_ = eta( By_, al0_, lam0_, prob)
    layers.append( ('LBelastic_net T=1',xhat_, (al0_, lam0_, B_, S_) ) )
    #pdb.set_trace()
    for t in range(1,T):
        al_ = tf.Variable( initial_lambda,name='al_{0}'.format(t) )
        lam_ = tf.Variable( initial_lambda,name='lam_{0}'.format(t) )
        B_ =  tf.Variable(B,dtype=tf.float32,name='B_{0}'.format(t) )
        By_ = tf.matmul(B_,prob.y_)
        S_ = tf.Variable( np.identity(N) - np.matmul(B,A),dtype=tf.float32,name='S_{0}'.format(t) )
        xhat_ = eta( tf.matmul(S_,xhat_) + By_, al_, lam_, prob)
        layers.append( ('LBelastic_net T='+str(t+1),xhat_,(al_, lam_, B_, S_)) )

    return layers

def build_LBFastelastic_net(prob,T,initial_lambda=.1):
    """
    Builds a LBFastelastic_net network to infer x from prob.y_ = matmul(prob.A,x) + AWGN
    prob            - is a TFGenerator which contains problem parameters and def of how to generate training data
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

    al0_ = tf.Variable( initial_lambda,name='al_0')
    lam0_ = tf.Variable( initial_lambda,name='lam_0')
    xhat_ = eta( By_, al0_, lam0_, prob)
    tk = (1+np.sqrt(1+4*1**2))*2**(-1)
    z_ = xhat_
    layers.append( ('LBFastelastic_net T=1',xhat_, (al0_, lam0_) ) )
    #pdb.set_trace()
    for t in range(1,T):
        t_prev = tk
        xhat_prev_ = xhat_ 
        al_ = tf.Variable( initial_lambda,name='al_{0}'.format(t) )
        lam_ = tf.Variable( initial_lambda,name='lam_{0}'.format(t) )
        xhat_ = eta( tf.matmul(S_,z_) + By_, al_, lam_, prob)
        tk = (1+np.sqrt(1+4*t_prev**2))*2**(-1)
        z_ = xhat_ + (t_prev-1)*(tk)**(-1)*(xhat_-xhat_prev_)
        layers.append( ('LBFastelastic_net T='+str(t+1),xhat_,(al_, lam_)) )

    return layers

def build_LADMM(prob, T, initial_lambda=.1, initial_rho=.1):
    layers = []
    m,n = np.shape(prob.A)
    prox = block_soft_threshold
    initial_lambda = np.array(initial_lambda).astype(np.float32)
    lam_ = tf.Variable(initial_lambda, name='lam_0')
    rho_ = tf.Variable(initial_rho, name='rho_0')
    A_ = tf.Variable(prob.A, trainable = True)
    B_ = tf.matmul(tf.transpose(A_),A_)+rho_*tf.eye(n)

    x1_ = tf.matmul(tf.linalg.inv(B_),tf.matmul(tf.transpose(A_), prob.y_))
    z1_ = prox(x1_, lam_,prob)
    u1_ = x1_
    layers.append(('LADMM T=1', z1_, (lam_, rho_, A_)))
    for t in range(1, T):
        lam_ = tf.Variable(initial_lambda, name=f'lam_{t}')
        rho_ = tf.Variable(initial_rho, name=f'rho_{t}')
        A_ = tf.Variable(prob.A, trainable=True)
        x1_ = tf.matmul(tf.linalg.inv(B_), (tf.matmul(tf.transpose(A_), prob.y_)) + rho_ * (z1_ - u1_))
        z1_ = prox(x1_ + u1_, lam_,prob)
        u1_ = u1_ + x1_ - z1_
        layers.append(('LADMM T=' + str(t+1), z1_, (lam_,rho_, A_)))

    return layers
