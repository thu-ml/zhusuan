#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Logistic-normal topic models using Monte-Carlo EM
Sparse implementation, O(DKL + KV)
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import sys
import os
import time

import tensorflow as tf
from six.moves import range
import numpy as np
import zhusuan as zs

from examples import conf
from examples.utils import dataset


# corresponding to eta in LDA. Larger log_delta leads to sparser topic.
log_delta = 10.0


@zs.reuse('model')
def lntm(observed, D, K, V, eta_mean, eta_logstd):
    with zs.BayesianNet(observed=observed) as model:
        eta = zs.Normal('eta',
                        tf.tile(tf.expand_dims(eta_mean, 0), [D, 1]),
                        tf.tile(tf.expand_dims(eta_logstd, 0), [D, 1]),
                        group_event_ndims=1)
        beta = zs.Normal('beta', tf.zeros([K, V]), tf.ones([K, V]) * log_delta,
                         group_event_ndims=1)
    return model


def lil_to_coo(X):
    ds = []
    vs = []
    cs = []
    for d, row in enumerate(X):
        for v, c in row:
            ds.append(d)
            vs.append(v)
            cs.append(c)
    return ds, vs, cs


if __name__ == "__main__":
    tf.set_random_seed(1237)

    # Load nips dataset
    data_name = 'nips'
    data_path = os.path.join(conf.data_dir, data_name + '.pkl.gz')
    print('Loading data...')
    X, vocab = dataset.load_uci_bow_sparse(data_name, data_path)
    print('Finished.')
    X_train = X[:1200]
    X_test = X[1200:]

    # Define model training/evaluation parameters
    D = 100
    K = 100
    V = len(vocab)
    num_e_steps = 5
    hmc = zs.HMC(step_size=1e-3, n_leapfrogs=20, adapt_step_size=True,
                 target_acceptance_rate=0.6)
    epoches = 100
    learning_rate_0 = 1.0
    t0 = 10

    T = 0
    for row in X_train:
        for _, c in row:
            T += c

    iters = len(X_train) // D
    Eta = np.zeros((len(X_train), K), dtype=np.float32)
    Eta_mean = np.zeros(K, dtype=np.float32)
    Eta_logstd = np.zeros(K, dtype=np.float32)

    # Build the computation graph
    ds = tf.placeholder(tf.int32, shape=[None], name='ds')
    vs = tf.placeholder(tf.int32, shape=[None], name='vs')
    cs = tf.placeholder(tf.float32, shape=[None], name='cs')
    eta_mean = tf.placeholder(tf.float32, shape=[K], name='eta_mean')
    eta_logstd = tf.placeholder(tf.float32, shape=[K], name='eta_logstd')
    eta = tf.Variable(tf.zeros([D, K]), name='eta')
    eta_ph = tf.placeholder(tf.float32, shape=[D, K], name='eta_ph')
    beta = tf.Variable(tf.zeros([K, V]), name='beta')
    phi = tf.nn.softmax(beta)
    init_eta = tf.assign(eta, tf.zeros([D, K]))
    init_eta_ph = tf.assign(eta, eta_ph)

    def joint_obj(observed):
        model = lntm(observed, D, K, V, eta_mean, eta_logstd)
        # [D], [K], [K]
        log_p_eta, log_p_beta = \
            model.local_log_prob(['eta', 'beta'])

        theta = tf.nn.softmax(observed['eta'])
        phi = tf.nn.softmax(observed['beta'])
        phi_t = tf.transpose(phi)

        thetas = tf.gather(theta, ds)
        phis = tf.gather(phi_t, vs)
        pred = tf.log(tf.reduce_sum(thetas * phis, -1))
        indices = tf.expand_dims(ds, 1)
        log_px = tf.scatter_nd(indices, cs * pred, shape=[D])

        return log_p_eta, log_p_beta, log_px

    def e_obj(observed):
        log_p_eta, _, log_px = joint_obj(observed)
        return log_p_eta + log_px

    lp_eta, lp_beta, lp_x = joint_obj({'eta': eta, 'beta': beta})
    log_likelihood = tf.reduce_sum(lp_x)
    log_joint = tf.reduce_sum(lp_beta) + log_likelihood
    sample_op = hmc.sample(e_obj, {'beta': beta}, {'eta': eta})

    learning_rate_ph = tf.placeholder(tf.float32, shape=[], name='lr')
    optimizer = tf.train.AdamOptimizer(learning_rate_ph)
    infer = optimizer.minimize(-log_joint, var_list=[beta])

    params = tf.trainable_variables()
    for i in params:
        print(i.name, i.get_shape())

    # Run the inference
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(1, epoches + 1):
            time_epoch = -time.time()
            learning_rate = learning_rate_0 / (t0 + epoch) * t0
            perm = list(range(len(X_train)))
            np.random.shuffle(perm)
            X_train = [X_train[p] for p in perm]
            Eta = Eta[perm, :]
            lls = []
            accs = []
            for t in range(iters):
                x_batch = X_train[t * D: (t + 1) * D]
                old_eta = Eta[t * D:(t + 1) * D, :]
                ds_batch, vs_batch, cs_batch = lil_to_coo(x_batch)

                # E step
                sess.run(init_eta_ph, feed_dict={eta_ph: old_eta})
                for j in range(num_e_steps):
                    new_eta, _, _, _, _, _, acc, _ = sess.run(
                        sample_op,
                        feed_dict={ds: ds_batch, vs: vs_batch, cs: cs_batch,
                                   eta_mean: Eta_mean,
                                   eta_logstd: Eta_logstd})
                    accs.append(acc)
                    # Store eta for the persistent chain
                    if j + 1 == num_e_steps:
                        Eta[t * D:(t + 1) * D, :] = new_eta[0]

                # M step
                _, ll = sess.run(
                    [infer, log_likelihood],
                    feed_dict={ds: ds_batch, vs: vs_batch,
                               cs: cs_batch,
                               eta_mean: Eta_mean,
                               eta_logstd: Eta_logstd,
                               learning_rate_ph: learning_rate * t0 / (
                                   t0 + epoch)})
                lls.append(ll)

            # Update hyper-parameters
            Eta_mean = np.mean(Eta, axis=0)
            Eta_logstd = np.log(np.std(Eta, axis=0) + 1e-6)

            time_epoch += time.time()
            print('Epoch {} ({:.1f}s): Perplexity = {:.2f}, acc = {:.3f}, '
                  'eta mean = {:.2f}, logstd = {:.2f}'
                  .format(epoch, time_epoch, np.exp(-np.sum(lls) / T),
                          np.mean(accs), np.mean(Eta_mean),
                          np.mean(Eta_logstd)))

        # Output topics
        p = sess.run(phi)
        for k in range(K):
            rank = zip(list(p[k, :]), range(V))
            rank.sort()
            rank.reverse()
            sys.stdout.write('Topic {}, eta mean = {:.2f} stdev = {:.2f}: '
                             .format(k, Eta_mean[k], np.exp(Eta_logstd[k])))
            for i in range(10):
                sys.stdout.write(vocab[rank[i][1]] + ' ')
            sys.stdout.write('\n')
