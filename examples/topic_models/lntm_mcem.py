#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Logistic-normal topic models using Monte-Carlo EM
Dense implementation, O(DKV)
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
def lntm(observed, n_chains, D, K, V, eta_mean, eta_logstd):
    with zs.BayesianNet(observed=observed) as model:
        D_multiple = tf.stack([D, 1])
        n_chains_multiple = tf.stack([n_chains, 1, 1])
        eta = zs.Normal('eta',
                        tf.tile(tf.expand_dims(
                            tf.tile(tf.expand_dims(eta_mean, 0), D_multiple),
                            0), n_chains_multiple),
                        logstd=eta_logstd,
                        group_ndims=1)
        beta = zs.Normal('beta', tf.zeros([K, V]), logstd=log_delta,
                         group_ndims=1)
    return model


if __name__ == "__main__":
    tf.set_random_seed(1237)

    # Load nips dataset
    data_name = 'nips'
    data_path = os.path.join(conf.data_dir, data_name + '.pkl.gz')
    X, vocab = dataset.load_uci_bow(data_name, data_path)
    X_train = X[:1200, :]
    X_test = X[1200:, :]

    # Define model training/evaluation parameters
    D = 100
    K = 100
    V = X_train.shape[1]
    n_chains = 1

    num_e_steps = 5
    hmc = zs.HMC(step_size=1e-3, n_leapfrogs=20, adapt_step_size=True,
                 target_acceptance_rate=0.6)
    epochs = 100
    learning_rate_0 = 1.0
    t0 = 10

    # Padding
    rem = D - X_train.shape[0] % D
    if rem < D:
        X_train = np.vstack((X_train, np.zeros((rem, V))))

    T = np.sum(X_train)
    iters = X_train.shape[0] // D
    Eta = np.zeros((n_chains, X_train.shape[0], K), dtype=np.float32)
    Eta_mean = np.zeros(K, dtype=np.float32)
    Eta_logstd = np.zeros(K, dtype=np.float32)

    # Build the computation graph
    x = tf.placeholder(tf.float32, shape=[D, V], name='x')
    eta_mean = tf.placeholder(tf.float32, shape=[K], name='eta_mean')
    eta_logstd = tf.placeholder(tf.float32, shape=[K], name='eta_logstd')
    eta = tf.Variable(tf.zeros([n_chains, D, K]), name='eta')
    eta_ph = tf.placeholder(tf.float32, shape=[n_chains, D, K], name='eta_ph')
    beta = tf.Variable(tf.zeros([K, V]), name='beta')
    phi = tf.nn.softmax(beta)
    init_eta_ph = tf.assign(eta, eta_ph)

    D_ph = tf.placeholder(tf.int32, shape=[], name='D_ph')
    n_chains_ph = tf.placeholder(tf.int32, shape=[], name='n_chains_ph')

    def joint_obj(observed):
        model = lntm(observed, n_chains_ph, D_ph, K, V, eta_mean, eta_logstd)

        log_p_eta, log_p_beta = \
            model.local_log_prob(['eta', 'beta'])

        theta = tf.nn.softmax(observed['eta'])
        theta = tf.reshape(theta, [-1, K])
        phi = tf.nn.softmax(observed['beta'])
        pred = tf.matmul(theta, phi)
        pred = tf.reshape(pred, tf.stack([n_chains_ph, D_ph, V]))
        x = tf.expand_dims(observed['x'], 0)
        log_px = tf.reduce_sum(x * tf.log(pred), -1)

        # Shape:
        # log_p_eta, log_px: [n_chains, D]
        # log_p_beta: [K]
        return log_p_eta, log_p_beta, log_px

    def e_obj(observed):
        log_p_eta, _, log_px = joint_obj(observed)
        return log_p_eta + log_px

    lp_eta, lp_beta, lp_x = joint_obj({'x': x, 'eta': eta, 'beta': beta})
    log_likelihood = tf.reduce_sum(tf.reduce_mean(lp_x, axis=0), axis=0)
    log_joint = tf.reduce_sum(lp_beta) + log_likelihood
    sample_op, hmc_info = hmc.sample(
        e_obj, observed={'x': x, 'beta': beta}, latent={'eta': eta})

    learning_rate_ph = tf.placeholder(tf.float32, shape=[], name='lr')
    optimizer = tf.train.AdamOptimizer(learning_rate_ph)
    infer = optimizer.minimize(-log_joint, var_list=[beta])

    params = tf.trainable_variables()
    for i in params:
        print(i.name, i.get_shape())

    # Below is the evaluation part.
    # Variables whose name starts with '_' is only used in the evaluation part,
    # to be distinguished from those variables used in the training part above.

    _D = X_test.shape[0]
    _T = np.sum(X_test)
    _n_chains = 25
    _n_temperatures = 1000

    _x = tf.placeholder(tf.float32, shape=[_D, V], name='x')
    _eta = tf.Variable(tf.zeros([_n_chains, _D, K]), name='eta')

    def _log_prior(observed):
        log_p_eta, _, _ = joint_obj(observed)
        return log_p_eta

    _prior_samples = {'eta': lntm({}, _n_chains, _D, K, V,
                      eta_mean, eta_logstd).outputs('eta')}

    _hmc = zs.HMC(step_size=0.01, n_leapfrogs=20, adapt_step_size=True,
                  target_acceptance_rate=0.6)

    _ais = zs.evaluation.AIS(_log_prior, e_obj, _prior_samples, _hmc,
                             observed={'x': _x, 'beta': beta},
                             latent={'eta': _eta},
                             n_chains=_n_chains,
                             n_temperatures=_n_temperatures)

    # -------------------

    # Run the inference
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(1, epochs + 1):
            time_epoch = -time.time()
            learning_rate = learning_rate_0 / (t0 + epoch) * t0
            perm = list(range(X_train.shape[0]))
            np.random.shuffle(perm)
            X_train = X_train[perm, :]
            Eta = Eta[:, perm, :]
            lls = []
            accs = []
            for t in range(iters):
                x_batch = X_train[t * D: (t + 1) * D]
                old_eta = Eta[:, t * D:(t + 1) * D, :]

                # E step
                sess.run(init_eta_ph, feed_dict={eta_ph: old_eta})
                for j in range(num_e_steps):
                    _, new_eta, acc = sess.run(
                        [sample_op, hmc_info.samples['eta'],
                         hmc_info.acceptance_rate],
                        feed_dict={x: x_batch,
                                   eta_mean: Eta_mean,
                                   eta_logstd: Eta_logstd,
                                   D_ph: D,
                                   n_chains_ph: n_chains})
                    accs.append(acc)
                    # Store eta for the persistent chain
                    if j + 1 == num_e_steps:
                        Eta[:, t * D:(t + 1) * D, :] = new_eta

                # M step
                _, ll = sess.run(
                    [infer, log_likelihood],
                    feed_dict={x: x_batch,
                               eta_mean: Eta_mean,
                               eta_logstd: Eta_logstd,
                               learning_rate_ph: learning_rate * t0 / (
                                   t0 + epoch),
                               D_ph: D,
                               n_chains_ph: n_chains})
                lls.append(ll)

            # Update hyper-parameters
            Eta_mean = np.mean(Eta, axis=(0, 1))
            Eta_logstd = np.log(np.std(Eta, axis=(0, 1)) + 1e-6)

            time_epoch += time.time()
            print('Epoch {} ({:.1f}s): Perplexity = {:.2f}, acc = {:.3f}, '
                  'eta mean = {:.2f}, logstd = {:.2f}'
                  .format(epoch, time_epoch, np.exp(-np.sum(lls) / T),
                          np.mean(accs), np.mean(Eta_mean),
                          np.mean(Eta_logstd)))

        # Run AIS
        time_ais = -time.time()

        ll_lb = _ais.run(sess, feed_dict={_x: X_test,
                                          eta_mean: Eta_mean,
                                          eta_logstd: Eta_logstd,
                                          D_ph: _D,
                                          n_chains_ph: _n_chains})

        time_ais += time.time()

        print('>> Test perplexity (AIS) ({:.1f}s)\n'
              '>> loglikelihood lower bound = {}\n'
              '>> perplexity upper bound = {}'
              .format(time_ais, ll_lb, np.exp(-ll_lb * _D / _T)))

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
