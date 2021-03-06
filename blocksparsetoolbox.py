import numpy as np
import scipy.io as sio
import numpy.linalg as la
import pdb
from scipy import linalg

# Toolbox for block sparse computation
# x is a n dim block sparse vector with M blocks of length d and N = M*d
# y is a vector from R^L, L=R*d
# D is an LxN Matrix

def l21norm(x, M, d):
    x = np.reshape(x, (M, d))
    r = np.sum(np.abs(x), axis=1)
    return np.sum(r)

def mutual_block_coherence(W,D, M, d):
    # M y Dimension, d length of blocks
    r = np.zeros((M, M))
    for l in range(0, M):
        for k in range(0, M):
            if l != k:
                r[l, k] = np.linalg.norm(W[:, l * d: (l + 1) * d].T @ D[:, k * d: (k + 1) * d], 2)
            else :
                r[l, k] = -np.inf
    return np.max(1 / d * r)

def block_coherence(D, M, d):
    r = np.zeros((M, M))
    for l in range(0, M):
        for k in range(0, M):
            if l!=k:
                r[l,k] = np.linalg.norm(D[:, l*d: (l+1)*d].T@D[:, k*d: (k+1)*d], 2)
            else :
                r[l, k] = -np.inf
    return np.max(1/d*r)


def C_w(W, M, d):
    r = np.zeros(M)
    for i in range(0, M):
        r[i] = np.linalg.norm(W[:, d * i: d * (i + 1)], 2)
    return np.max(r)

def nmse(xhat,x):
    normx=la.norm(x, ord='fro')**2
    return la.norm((xhat-x), ord='fro')**2/normx

def nmse_db(xhat,x):
    return 10*np.log10(nmse(xhat,x))

def compute_W_Lagrange(D,L,d):
    W = np.zeros(D.shape)
    m,n = D.shape
    for i in range(0,L):
        #pdb.set_trace()
        A1 = np.concatenate((2*D@D.T, D[:, i*d: (i+1)*d]), axis=1)
        A2 = np.concatenate((D[:, i*d: (i+1)*d].T, np.zeros((d, d))),axis=1)
        A = np.concatenate((A1,A2), axis = 0)
        #pdb.set_trace()
        rhs = np.concatenate((np.zeros((m,d)), np.eye(d)))
        X = linalg.pinv(A)@rhs
        W[:,i * d:(i + 1) * d] = X[0:m,:]

    return W

#2, 2*kron(eye(B), D*D'), kron(eye(B),D(:,(k-1)*B+1:k*B)
#cat(2, kron(eye(B),D(:,(k-1)*B+1:k*B)'), zeros(B^2, B^2));
#cat(1, zeros(B*m,1),reshape(eye(B),[],1));

def sparsity(x, M, d):
    s=0
    for l in range(0,M):
        if np.linalg.norm(x[l*d: (l+1)*d], 2)!=0:
            s = s+1
        else:
            s = s+0
    return s
