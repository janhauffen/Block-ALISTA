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
#import cvxpy as cp

def SolveViaFFT(D):
    # In the setting of a circular convolution MMV Problem we have an easy way to compute the analytical weight matrix
    m,n = D.shape
    u,s,v = np.linalg.svd(D)
    s[np.linalg.matrix_rank(D):n] = 0
    k_hat=(1/s)/(np.linalg.matrix_rank(D)/n)
    for i in range(np.linalg.matrix_rank(D),n):
        k_hat[i]=0
    B_5 = np.real(u@np.diag(k_hat)@v)
    return B_5

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
    r_ = K.reshape(r_, pool_r) 
    pool_shape2 = tf.stack([B * L, shape[1]])

    Xnew_ = tf.multiply(Xnew_, r_)

    return K.reshape(Xnew_, pool_shape2)

def block_soft_threshold_elastic_net(X_, al_, la_, prob):
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
    r_ = K.reshape(r_, pool_r)
    pool_shape2 = tf.stack([B * L, shape[1]])

    Xnew_ = tf.multiply(Xnew_, r_)*(1+la_)**(-1)

    return K.reshape(Xnew_, pool_shape2)

def tf_kron(a,b):#https://datascience.stackexchange.com/questions/47286/kronecker-product-using-sparse-matrices-tensors-in-tensorflow
    a_shape = [a.shape[0].value,a.shape[1].value]
    b_shape = [b.shape[0].value,b.shape[1].value]
    return tf.reshape(tf.reshape(a,[a_shape[0],1,a_shape[1],1])*tf.reshape(b,[1,b_shape[0],1,b_shape[1]]),[a_shape[0]*b_shape[0],a_shape[1]*b_shape[1]])


def build_LBISTA(prob,W, T,initial_lambda=.1):
    """
    Builds a LBISTA network to infer x from prob.y_ = matmul(prob.A,x) + AWGN
    prob            - is a TFGenerator which contains problem parameters and def of how to generate training data
    Return a list of layer info (name,xhat_,newvars)
     name : description, e.g. 'LISTA T=1'
     xhat_ : that which approximates x_ at some point in the algorithm
     newvars : a tuple of layer-specific trainable variables
    """
    blocksoft=block_soft_threshold
    layers = []
    A = prob.A
    M,N = A.shape
    
    B = W.T
    B_s =  tf.Variable(B, dtype=tf.float32 ,name='B_0')
   
    S_s = tf.Variable( np.identity(prob.A_s.shape[1]) - np.matmul(B,prob.A_s),dtype=tf.float32,name='S_0')
    S_ = tf_kron(S_s, tf.eye(prob.B))
    B_ = tf_kron(B_s, tf.eye(prob.B))
    By_ = tf.matmul(B_,prob.y_)
    
    initial_lambda = np.array(initial_lambda).astype(np.float32)

    lam0_ = tf.Variable( initial_lambda,name='lam_0')
    xhat_ = blocksoft( By_, lam0_, prob)
    layers.append( ('LBISTA T=1',xhat_, (lam0_,B_s, S_s) ) )
    for t in range(1,T):
        lam_ = tf.Variable( initial_lambda,name='lam_{0}'.format(t) )
        S_ = tf_kron(S_s, tf.eye(prob.B))
        B_ = tf_kron(B_s, tf.eye(prob.B))
        xhat_ = blocksoft( tf.matmul(S_,xhat_) + tf.matmul(B_,prob.y_), lam_ , prob)
        layers.append( ('LBISTA T='+str(t+1),xhat_,(lam_, B_s, S_s)) )

    return layers


def build_LBISTA_untied(prob, W, T, initial_lambda=.1):
    """
    Builds a LBISTA network to infer x from prob.y_ = matmul(prob.A,x) + AWGN
    prob            - is a TFGenerator which contains problem parameters and def of how to generate training data
    Return a list of layer info (name,xhat_,newvars)
     name : description, e.g. 'LISTA T=1'
     xhat_ : that which approximates x_ at some point in the algorithm

     newvars : a tuple of layer-specific trainable variables
    """
    blocksoft=block_soft_threshold
    layers = []
    A = prob.A
    M,N = A.shape
    
    B = W.T
    B_s =  tf.Variable(B, dtype=tf.float32 ,name='B_0')
    B_ = tf_kron(B_s, tf.eye(prob.B))
    By_ = tf.matmul(B_,prob.y_)
    S_s = tf.Variable( np.identity(prob.A_s.shape[1]) - np.matmul(B,prob.A_s),dtype=tf.float32,name='S_0')
    S_ = tf_kron(S_s, tf.eye(prob.B))

    initial_lambda = np.array(initial_lambda).astype(np.float32)

    lam0_ = tf.Variable( initial_lambda,name='lam_0')
    xhat_ = blocksoft( By_, lam0_, prob)
    layers.append( ('LBISTA T=1',xhat_, (lam0_,B_s, S_s) ) )
    #pdb.set_trace()
    for t in range(1,T):
        B_s =  tf.Variable(B, dtype=tf.float32 ,name='B_{0}'.format(t) )
        B_ = tf_kron(B_s, tf.eye(prob.B))
        By_ = tf.matmul(B_,prob.y_)
        S_s = tf.Variable( np.identity(prob.A_s.shape[1]) - np.matmul(B,prob.A_s),dtype=tf.float32,name='S_{0}'.format(t) )
        S_ = tf_kron(S_s, tf.eye(prob.B))
        lam_ = tf.Variable( initial_lambda,name='lam_{0}'.format(t) )
        xhat_ = blocksoft( tf.matmul(S_,xhat_) + tf.matmul(B_,prob.y_), lam_ , prob)
        layers.append( ('LBISTA T='+str(t+1),xhat_,(lam_,B_s, S_s)) )

    return layers

def build_LBISTA_CPSS(prob, W, T, initial_lambda=.1, initial_gamma=1):
    """
    Builds a LBISTA_CPSS network to infer x from prob.y_ = matmul(prob.A,x) + AWGN
    prob            - is a TFGenerator which contains problem parameters and def of how to generate training data
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
    
    W_s =  tf.Variable(W.T, dtype=tf.float32 ,name='W_0')
    W_ = tf_kron(W_s, tf.eye(prob.B))
    
    initial_lambda = np.array(initial_lambda).astype(np.float32)
    lam0_ = tf.Variable(initial_lambda, name='lam_0')
    xhat_ = blocksoft(tf.matmul(tf.math.abs(gamma0_)*W_, prob.y_), lam0_, prob)
    layers.append(('LBISTA_CPSS T=1', xhat_, (lam0_,gamma0_, W_s)))
    # pdb.set_trace()
    for t in range(1, T):
        lam_ = tf.Variable(initial_lambda, name='lam_{0}'.format(t))
        W_s =  tf.Variable(W.T, dtype=tf.float32 , name='W_{0}'.format(t))
        W_ = tf_kron(W_s, tf.eye(prob.B))
        gamma_ = tf.Variable(gamma, dtype=tf.float32, name='gamma_{0}'.format(t))
        xhat_ = blocksoft(xhat_ - tf.matmul(tf.math.abs(gamma_)*W_, tf.matmul(prob.A_, xhat_) - prob.y_), lam_, prob)
        layers.append(('LBISTA_CPSS T=' + str(t + 1), xhat_, (lam_,gamma_, W_s)))

    return layers

def build_ALBISTA(prob, W, T, initial_lambda=.1, initial_gamma=1):
    """
    Builds a ALBISTA network to infer x from prob.y_ = matmul(prob.A,x) + AWGN
    prob            - is a TFGenerator which contains problem parameters and def of how to generate training data
    W               - W is the analytical weight matrix, check Optimize_GenBlockCoheherence.ipynb to look up different computing
                      strategies
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
        layers.append(('BALISTA T=' + str(t + 1), xhat_, (lam_, gamma_)))

    return layers

def build_TiBLISTA(prob, W, T, initial_lambda=.1, initial_gamma=1):
    """
    Builds a TiBLISTA network to infer x from prob.y_ = matmul(prob.A,x) + AWGN
    prob            - is a TFGenerator which contains problem parameters and def of how to generate training data
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
    W_s =  tf.Variable(W.T, dtype=tf.float32 ,name='W_0')
    W_ = tf_kron(W_s, tf.eye(prob.B))
    
    initial_lambda = np.array(initial_lambda).astype(np.float32)
    lam0_ = tf.Variable(initial_lambda, name='lam_0')
    xhat_ = blocksoft(tf.matmul(tf.math.abs(gamma0_)*W_, prob.y_), lam0_, prob)
    layers.append(('TiBLISTA T=1', xhat_, (lam0_,gamma0_,W_s)))
    # pdb.set_trace()
    for t in range(1, T):
        lam_ = tf.Variable(initial_lambda, name='lam_{0}'.format(t))
        gamma_ = tf.Variable(gamma, dtype=tf.float32, name='gamma_{0}'.format(t))
        
        xhat_ = blocksoft(xhat_ - tf.matmul(tf.math.abs(gamma_)*W_, tf.matmul(prob.A_, xhat_) - prob.y_), lam_, prob)
        
        layers.append(('TiBLISTA T=' + str(t + 1), xhat_, (lam_,gamma_,W_s)))

    return layers

def build_LBFISTA(prob,T,initial_lambda=.1):
    """
    Builds a LBFISTA network to infer x from prob.y_ = matmul(prob.A,x) + AWGN
    prob            - is a TFGenerator which contains problem parameters and def of how to generate training data
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
    layers.append( ('LBFISTA T=1',xhat_, (lam0_,B_, S_) ) )
    #pdb.set_trace()
    for t in range(1,T):
        t_prev = tk
        xhat_prev_ = xhat_ 
        lam_ = tf.Variable( initial_lambda,name='lam_{0}'.format(t) )
        xhat_ = blocksoft( tf.matmul(S_,z_) + By_, lam_, prob)
        tk = (1+np.sqrt(1+4*t_prev**2))*2**(-1)
        z_ = xhat_ + (t_prev-1)*(tk)**(-1)*(xhat_-xhat_prev_)
        layers.append( ('LBFISTA T='+str(t+1),xhat_,(lam_,B_, S_)) )

    return layers


def build_CircALBISTA(prob, W, T, initial_lambda=.1, initial_gamma=1):
    """
    Builds a LISTA network to infer x from prob.y_ = matmul(prob.A,x) + AWGN
    prob            - is a TFGenerator which contains problem parameters and def of how to generate training data
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

    W_ = tf.Variable(W.T, dtype=tf.float32, trainable = False)
    
    initial_lambda = np.array(initial_lambda).astype(np.float32)
    lam0_ = tf.Variable(initial_lambda, name='lam_0')
    xhat_ = blocksoft(tf.matmul(tf.math.abs(gamma0_)*W_, prob.y_), lam0_, prob)
    layers.append(('CircALBISTA T=1', xhat_, (lam0_, gamma0_)))
    # pdb.set_trace()
    for t in range(1, T):
        lam_ = tf.Variable(initial_lambda, name='lam_{0}'.format(t))
        gamma_ = tf.Variable(gamma, dtype=tf.float32, name='gamma_{0}'.format(t))
        xhat_ = blocksoft(xhat_ - tf.matmul(tf.math.abs(gamma_)*W_, tf.matmul(prob.A_, xhat_) - prob.y_), lam_, prob)
        layers.append(('CircALBISTA T=' + str(t + 1), xhat_, (lam_, gamma_)))

    return layers

def newfft(X_):
    # Circumvent problem for placeholder with dimension (n, batch) -> (batch, n)
    return (tf.signal.rfft(tf.transpose(X_)))

def newifft(X_):
    # Circumvent problem for placeholder with dimension (batch, n) -> (n, batch)
    return tf.transpose(tf.signal.irfft((X_)))


def build_FFT_ALBISTA_old(prob, W, T, initial_lambda=.1, initial_gamma=1):
    """
    Builds a LISTA network to infer x from prob.y_ = matmul(prob.A,x) + AWGN
    prob            - is a TFGenerator which contains problem parameters and def of how to generate training data
    Return a list of layer info (name,xhat_,newvars)
     name : description, e.g. 'LISTA T=1'
     xhat_ : that which approximates x_ at some point in the algorithm
     newvars : a tuple of layer-specific trainable variables
    """
    blocksoft = block_soft_threshold
    layers = []
    A = prob.A[:,0]
    A = tf.signal.rfft(A)
    gamma = initial_gamma
    gamma0_ = tf.Variable(gamma, dtype=tf.float32, name='gamma_0')

    W_ = tf.Variable(W.T[:,0], dtype=tf.float32, trainable = False)
    W_ = tf.signal.rfft(W_)
    
    initial_lambda = np.array(initial_lambda).astype(np.float32)
    lam0_ = tf.Variable(initial_lambda, name='lam_0')
    #pdb.set_trace()
    #xhat_ = blocksoft(tf.matmul(tf.math.abs(gamma0_)*W_, prob.y_), lam0_, prob)
    xhat_ = blocksoft(tf.math.abs(gamma0_)*newifft(W_*newfft(prob.y_)), lam0_, prob)
    layers.append(('CircALBISTA T=1', xhat_, (lam0_, gamma0_)))
    e = np.zeros(A.shape)
    e[0] = 1
    
    # pdb.set_trace()
    for t in range(1, T):
        lam_ = tf.Variable(initial_lambda, name='lam_{0}'.format(t))
        gamma_ = tf.Variable(gamma, dtype=tf.float32, name='gamma_{0}'.format(t))
        #xhat_ = blocksoft(xhat_ - tf.matmul(tf.math.abs(gamma_)*W_, tf.matmul(prob.A_, xhat_) - prob.y_), lam_, prob)
        #f = tf.signal.rfft(e-tf.math.abs(gamma_)*tf.signal.irfft(W_*A))
        #xhat_ = newifft(f*newfft(xhat_)) + newifft(tf.signal.rfft(W_)*newfft(prob.y_))
        xhat_ = xhat_ - tf.math.abs(gamma_)*newifft(W_*newfft(newifft(A*newfft(xhat_))-prob.y_))
        xhat_ = blocksoft(xhat_, lam_, prob)
        layers.append(('CircALBISTA T=' + str(t + 1), xhat_, (lam_, gamma_)))

    return layers

def build_MMV_ALAMP(prob, W, T, initial_lambda=.1, initial_gamma=1):
    """
    Builds a LISTA network to infer x from prob.y_ = matmul(prob.A,x) + AWGN
    prob            - is a TFGenerator which contains problem parameters and def of how to generate training data
    Return a list of layer info (name,xhat_,newvars)
     name : description, e.g. 'LISTA T=1'
     xhat_ : that which approximates x_ at some point in the algorithm
     newvars : a tuple of layer-specific trainable variables
    """
    blocksoft = block_soft_threshold
    layers = []
    A = prob.A
    M,N = A.shape
    gamma = initial_gamma
    gamma0_ = tf.Variable(gamma, dtype=tf.float32, name='gamma_0', trainable = True)

    W_ = tf.Variable(W.T, dtype=tf.float32, trainable = False)
    
    initial_lambda = np.array(initial_lambda).astype(np.float32)
    lam0_ = tf.Variable(initial_lambda, name='lam_0')
    #delta = 0.01
    delta = tf.Variable(0.01, dtype=tf.float32, name='delta', trainable = False)
    d_eta = (blocksoft(tf.matmul(tf.math.abs(gamma0_)*W_, prob.y_) + delta, lam0_, prob) - blocksoft(tf.matmul(tf.math.abs(gamma0_)*W_, prob.y_) - delta, lam0_, prob)) / (2*delta)
    d_eta = tf.math.reduce_mean(d_eta, axis = 0)
    #bt_ = d_eta
    xhat_ = blocksoft(tf.matmul(tf.math.abs(gamma0_)*W_, prob.y_), lam0_, prob)
    #d_eta = tf.math.reduce_mean(d_eta, axis = 0) * N/M
    vt_ = prob.y_
    
    layers.append(('MMV ALAMP T=1', xhat_, (lam0_, gamma0_, delta)))
    #pdb.set_trace()
    for t in range(1, T):
        lam_ = tf.Variable(initial_lambda, name='lam_{0}'.format(t))
        gamma_ = tf.Variable(gamma, dtype=tf.float32, name='gamma_{0}'.format(t), trainable = True)
        delta = tf.Variable(0.01, dtype=tf.float32, name='delta_{0}'.format(t), trainable = False)
        d_eta = (blocksoft(xhat_ + tf.matmul(tf.math.abs(gamma_)*W_, vt_) + delta, lam0_, prob) - blocksoft(xhat_ + tf.matmul(tf.math.abs(gamma_)*W_, vt_) - delta, lam0_, prob)) / (2*delta)
        d_eta = tf.math.reduce_mean(d_eta, axis = 0)
        bt_ = d_eta
        vt_ = prob.y_ - tf.matmul(A, xhat_) + bt_ * vt_
        
        xhat_ = blocksoft(xhat_ + tf.matmul(tf.math.abs(gamma_)*W_, vt_), lam_, prob)
        layers.append(('MMV ALAMP T=' + str(t + 1), xhat_, (lam_, gamma_, delta)))

    return layers
