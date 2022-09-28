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

def compute_W_v1(A,M,d):
    #via pseudo inverse
    W = np.zeros(A.shape)
    for i in range(0, M):
        W[:, i*d:(i+1)*d] = np.linalg.pinv(A[:, i*d: (i+1)*d].T)
    return W

def compute_W_v2(A,M,d):
    #pseudo inverse via svd
    W = np.zeros(A.shape)
    for i in range(0,M):
        u,s,v = np.linalg.svd(A[:,i*d: (i+1)*d].T)
        m,n = A[:,i*d: (i+1)*d].T.shape
        r = s.shape[0]
        if n==r:
            Sigma1 = np.diag(s**(-1))
            Sigma2 = np.zeros((n-r,m))
            Sigma = np.concatenate((Sigma1, Sigma2), axis=0)
        if n!=r:
            Sigma1 = np.concatenate((np.diag(s**(-1)), np.zeros((r,m-r))), axis=1)
            Sigma2 = np.zeros((n-r,m))
            Sigma = np.concatenate((Sigma1, Sigma2), axis=0)
        W[:,i*d:(i+1)*d] = v@Sigma@u.T
    return W

def compute_W_v3(A,M,d):
    #via pseudo inverse
    W = np.zeros(A.shape)
    for i in range(0, M):
        W[:, i*d:(i+1)*d] = linalg.pinv(A[:, i*d: (i+1)*d].T)
    return W

def compute_W_v4(A,M,d):
    #via pseudo inverse
    W = np.zeros(A.shape)
    for i in range(0, M):
        W[:, i*d:(i+1)*d] = linalg.pinv2(A[:, i*d: (i+1)*d].T)
    return W

def compute_W_v5(A,M,d):
    # theoretically right...
    W = np.zeros(A.shape)
    for i in range(0, M):
        W[:, i*d:(i+1)*d] = np.linalg.inv(A[:, i*d: (i+1)*d]@A[:, i*d: (i+1)*d].T)@A[:, i*d: (i+1)*d]
    return W

def compute_W_v6(A,M,d):
    # works good with algorithm
    W = np.zeros(A.T.shape)
    for i in range(0, M):
        W[i*d:(i+1)*d,:] = np.linalg.inv(A[:, i*d: (i+1)*d].T@A[:, i*d: (i+1)*d])@A[:, i*d: (i+1)*d].T
    return W.T

def compute_W_Lagrange(D,M,d):
    # works for d = 1...
    W = np.zeros(D.shape)
    m,n = D.shape
    for i in range(0,M):
        #pdb.set_trace()
        A1 = np.concatenate((np.kron(np.eye(d), D@D.T), np.kron(np.eye(d), D[:, i*d: (i+1)*d])), axis=1)
        A2 = np.concatenate((np.kron(np.eye(d), D[:, i*d: (i+1)*d].T), np.zeros((d**2, d**2))),axis=1)
        A = np.concatenate((A1,A2), axis = 0)
        rhs = np.concatenate((np.zeros((d*m,1)), np.reshape(np.eye(d), (d**2,1))))
        X = linalg.pinv(A)@rhs
        W[:,i * d:(i + 1) * d] = np.reshape(X[0:d*m, :], (m,d))

    return W

def compute_W_Thm(D, n, d):
    W = np.zeros(D.shape)
    m,n = D.shape
    for i in range(0,n):
        #pdb.set_trace()
        K = (2*D@D.T)@(2*D@D.T)+D[:,i * d:(i + 1) * d]@D[:,i * d:(i + 1) * d].T
        K_p = linalg.inv(K)
        E = 2*D@D.T@D[:,i * d:(i + 1) * d]
        R = D[:,i * d:(i + 1) * d] - 2*D@D.T@K_p@E
        S = - D[:,i * d:(i + 1) * d].T@K_p@E
        L = R.T@R + S.T@S
        I = np.eye((L.T@L).shape[0])
        M = K_p@E@(I - linalg.inv(L)@L)
        H = linalg.inv(L)@S.T + (I - linalg.inv(L)@L)@linalg.inv(I + M.T@M)@(K_p@E).T@K_p@(D[:,i * d:(i + 1) * d]-E@linalg.inv(L)@S.T)
        W[:,i * d:(i + 1) * d] = K_p@(D[:,i * d:(i + 1) * d]-E@H)
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