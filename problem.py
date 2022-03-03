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

class Generator(object):
    def __init__(self, A, **kwargs):
        self.A = A
        M, N = A.shape
        vars(self).update(kwargs)
        self.x_ = tf.placeholder(tf.float32, (N, None), name='x')
        self.y_ = tf.placeholder(tf.float32, (M, None), name='y')


class TFGenerator(Generator):
    def __init__(self, **kwargs):
        Generator.__init__(self, **kwargs)

    def __call__(self, sess):
        "generates y,x pair for training"
        return sess.run((self.ygen_, self.xgen_))


def block_gaussian_trial(m=128, L=32, B=16, MC=1000, pnz=.1, SNR_dB=20):
    N = B * L  # N is the length of a the unknown block-sparse x
    A2 = np.random.normal(size=(m, N), scale=1.0 / math.sqrt(m)).astype(np.float32)
    # pdb.set_trace()
    A2 = sio.loadmat('small_normalized_D.mat')
    A2 = A2.get('D')
    A=A2.astype('float32')
    A_ = tf.constant(A, name='A')
    prob = TFGenerator(A=A, A_=A_, kappa=None, SNR=SNR_dB)
    #pdb.set_trace()
    # prob.A=A
    prob.name = 'block sparse, Gaussian A'
    prob.L = L
    prob.B = B
    prob.N = N
    prob.m = m
    # prob.SNR_dB = SNR_dB
    prob.pnz = pnz

    # Create tf vectors
    active_blocks_ = tf.to_float(tf.random_uniform((L, 1 , MC)) < pnz)
    ones_ = tf.ones([L, B, MC])

    product_ = tf.multiply(active_blocks_, ones_)
    xgen_ = tf.reshape(product_, [L * B, MC])
    xgen_ = tf.multiply(xgen_, tf.random_normal((N, MC), 0, 1))

    # you should probably change the way noise_var is calculated
    noise_var = pnz * N / m * math.pow(10., -SNR_dB / 10.)
    ygen_ = tf.matmul(A_, xgen_) + tf.random_normal((m, MC), stddev=math.sqrt(noise_var))
    #ygen_ = tf.matmul(A_, xgen_)

    active_blocks_val = (np.random.uniform(0, 1, (L, MC)) < pnz).astype(np.float32)
    active_entries_val = np.repeat(active_blocks_val, B, axis=0)
    xval = np.multiply(active_entries_val, np.random.normal(0, 1, (N, MC)))
    yval = np.matmul(A, xval) + np.random.normal(0, math.sqrt(noise_var), (m, MC))
    #yval = np.matmul(A,xval)

    prob.xgen_ = xgen_
    prob.ygen_ = ygen_
    prob.xval = xval
    prob.yval = yval
    prob.noise_var = noise_var
    prob.noise = np.random.normal(0, math.sqrt(noise_var), (m))

    return prob
