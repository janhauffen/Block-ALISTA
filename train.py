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

def load_trainable_vars(sess, filename):
    """load a .npz archive and assign the value of each loaded
    ndarray to the trainable variable whose name matches the
    archive key.  Any elements in the archive that do not have
    a corresponding trainable variable will be returned in a dict.
    """
    other = {}
    try:
        tv = dict([(str(v.name), v) for v in tf.trainable_variables()])
        for k, d in np.load(filename).items():
            if k in tv:
                print('restoring ' + k + ' is:' + str(d))
                sess.run(tf.assign(tv[k], d))
            else:
                other[k] = d
    except IOError:
        pass
    return other


def save_trainable_vars(sess, filename, **kwargs):
    """save a .npz archive in `filename`  with
    the current value of each variable in tf.trainable_variables()
    plus any keyword numpy arrays.
    """
    save = {}
    for v in tf.trainable_variables():
        save[str(v.name)] = sess.run(v)
    save.update(kwargs)
    np.savez(filename, **save)


def setup_training(layer_info, prob, trinit=1e-3, final_refine=0.1):
    # with tf.device('/device:GPU:0'):
    """ Given a list of layer info (name,xhat_,newvars),
    create an output list of training operations (name,xhat_,loss_,nmse_,trainop_ ).
    """
    losses_ = []
    nmse_ = []
    trainers_ = []

    maskX_ = getattr(prob, 'maskX_', 1)
    if maskX_ != 1:
        print('masking out inconsequential parts of signal x for nmse reporting')

    nmse_denom_ = tf.nn.l2_loss(prob.x_ * maskX_)

    tr_ = tf.Variable(trinit, name='tr', trainable=False)
    training_stages = []
    for name, xhat_, var_list in layer_info:
        loss_ = tf.nn.l2_loss(xhat_ - prob.x_)
        nmse_ = tf.nn.l2_loss((xhat_ - prob.x_) * maskX_) / nmse_denom_
        "sigma2_ = tf.reduce_mean(rvar_)"
        "sigma2_empirical_ = tf.reduce_mean((rhat_ - prob.x_)**2)"

        se_ = 2 * tf.nn.l2_loss(xhat_ - prob.x_)  # to get MSE, divide by / (L * N)
        #pdb.set_trace()
        #if name == 'LBISTA T=6':
        if var_list is not None:
            train_ = tf.train.AdamOptimizer(tr_).minimize(loss_, var_list=var_list)
            training_stages.append((name, xhat_, loss_, nmse_, se_, train_, var_list))
        
        #train2_ = tf.train.AdamOptimizer(tr_ * final_refine).minimize(loss_)
        #training_stages.append((name + ' final refine ' + str(final_refine), xhat_, loss_, nmse_, se_, train2_, ()))

    return training_stages


def do_training(training_stages, prob, savefile, ivl=10, maxit=100000, better_wait=5000):
    # with tf.device('/device:GPU:0'):
    """
    ivl:how often should we compute the nmse of the validation set?
    maxit: max number of training iterations
    better_wait:wait this many iterations for an nmse that is better than the prevoius best of the current training session
    """
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    print('norms xval:{xval:.7f} yval:{yval:.7f}'.format(xval=la.norm(prob.xval), yval=la.norm(prob.yval)))

    state = load_trainable_vars(sess, savefile)  # must load AFTER the initializer

    # must use this same Session to perform all training
    # if we start a new Session, things would replay and we'd be training with our validation set (no no)

    done = state.get('done', [])
    log = str(state.get('log', ''))
    nmse_his = []
    nmse_switch = [] #save index of switching to next layer...
    for name, xhat_, loss_, nmse_, se_, train_, var_list in training_stages:
        start = time.time()
        if name in done:
            print('Already did ' + name + '. Skipping.')
            continue
        if len(var_list):
            describe_var_list = 'extending ' + ','.join([v.name for v in var_list])
        else:
            describe_var_list = 'fine tuning all ' + ','.join([v.name for v in tf.trainable_variables()])

        print(name + ' ' + describe_var_list)
        nmse_history = []
        loss_history = []
        for i in range(maxit + 1):

            if i % ivl == 0:
                nmse = sess.run(nmse_, feed_dict={prob.y_: prob.yval, prob.x_: prob.xval})
                if np.isnan(nmse):
                    #pdb.set_trace()
                    raise RuntimeError('nmse is NaN')
                nmse_history = np.append(nmse_history, nmse)
                nmse_his = np.append(nmse_his, nmse)
                nmse_dB = 10 * np.log10(nmse)
                nmsebest_dB = 10 * np.log10(nmse_history.min())
                sys.stdout.write(
                    '\ri={i:<6d} nmse={nmse:.6f} dB (best={best:.6f})'.format(i=i, nmse=nmse_dB, best=nmsebest_dB))
                sys.stdout.flush()
                if i % (100 * ivl) == 0:
                    print('')
                    age_of_best = len(nmse_history) - nmse_history.argmin() - 1  # how long ago was the best nmse?
                    if age_of_best * ivl > better_wait:
                        nmse_switch.append(i)
                        break  # if it has not improved on the best answer for quite some time, then move along
            y, x = prob(sess)

            # if name == 'LBISTA T=1':
            #    pdb.set_trace() 

            train, xhat, loss = sess.run([train_, xhat_, loss_],
                                         feed_dict={prob.y_: y, prob.x_: x})  # hier fehler lam0=nan
            loss_history.append(loss)
            train
        done = np.append(done, name)

        end = time.time()
        time_log = 'Took me {totaltime:.3f} minutes, or {time_per_interation:.1f} ms per iteration'.format(
            totaltime=(end - start) / 60, time_per_interation=(end - start) * 1000 / i)
        print(time_log)
        log = log + '\n{name} nmse={nmse:.6f} dB in {i} iterations'.format(name=name, nmse=nmse_dB, i=i)
        # pdb.set_trace()

        state['done'] = done
        state['log'] = log
        save_trainable_vars(sess, savefile, **state)
        
    return sess, nmse_his, loss_history, nmse_switch