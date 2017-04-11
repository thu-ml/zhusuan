#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Logistic-normal topic models using Monte-Carlo EM
# Dense implementation, O(DKV)

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
# If we place a Gamma prior on precision, we should be able to sample it
# analytically.
log_delta = 10.0


@zs.reuse('model')
def lntm(observed, D, K, V):
    with zs.BayesianNet(observed=observed) as model:
        log_alpha = zs.Normal('log_alpha', 0., 0.)
        beta = zs.Normal('beta', tf.zeros([K, V]), tf.ones([K, V]) * log_delta)
        eta = zs.Normal('eta', tf.zeros([D, K]), tf.ones([D, K]) * log_alpha)
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
    V = X.shape[1]
    current_log_alpha = 0.0
    num_e_steps = 5
    hmc = zs.HMC(step_size=1e-3, n_leapfrogs=20, adapt_step_size=True,
                 target_acceptance_rate=0.6)
    epoches = 100
    learning_rate_0 = 0.1
    t0 = 50
    mh_stdev = 0.1
    num_mh = 10

    # Padding
    rem = D - X.shape[0] % D
    if rem < D:
        X = np.vstack((X, np.zeros((rem, V))))

    T = np.sum(X)
    iters = X.shape[0] // D
    Eta = np.zeros((X.shape[0], K), dtype=np.float32)

    # Build the computation graph
    x = tf.placeholder(tf.float32, shape=[D, V], name='x')
    log_alpha = tf.placeholder(tf.float32, name='log_alpha')
    eta = tf.Variable(tf.zeros([D, K]), name='eta')
    eta_ph = tf.Variable(tf.zeros([D, K]), name='eta_ph')
    beta = tf.Variable(tf.zeros([K, V]), name='beta')
    phi = tf.nn.softmax(beta)
    init_eta = tf.assign(eta, tf.zeros([D, K]))
    init_eta_ph = tf.assign(eta, eta_ph)


    def joint_obj(observed):
        model = lntm(observed, D, K, V)
        # [D, K], [K, V]
        log_p_eta, log_p_beta, log_p_alpha = \
            model.local_log_prob(['eta', 'beta', 'log_alpha'])

        theta = tf.nn.softmax(observed['eta'])
        phi = tf.nn.softmax(observed['beta'])
        pred = tf.matmul(theta, phi)

        # D
        log_px = tf.reduce_sum(observed['x'] * tf.log(pred), -1)

        return log_p_eta, log_p_beta, log_px, log_p_alpha


    def e_obj(observed):
        log_p_eta, _, log_px, _ = joint_obj(observed)
        return tf.reduce_sum(log_p_eta, -1) + log_px


    lp_eta, lp_beta, lp_x, lp_alpha = \
        joint_obj({'x': x, 'eta': eta, 'beta': beta,
                   'log_alpha': log_alpha})
    log_likelihood = tf.reduce_sum(lp_x)
    log_joint = tf.reduce_sum(lp_beta) + log_likelihood
    log_hyper_post = lp_alpha + tf.reduce_sum(lp_eta)

    # Optimize
    sample_op = hmc.sample(e_obj, {'x': x, 'beta': beta,
                                   'log_alpha': log_alpha},
                           {'eta': eta}, chain_axis=0)
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
            learning_rate = learning_rate_0 / (t0 + epoch) * t0

            time_epoch = -time.time()
            perm = list(range(X.shape[0]))
            np.random.shuffle(perm)
            X = X[perm, :]
            Eta = Eta[perm, :]
            lls = []
            accs = []
            for t in range(iters):
                x_batch = X[t * D: (t + 1) * D]
                old_eta = Eta[t * D:(t + 1) * D, :]
                # E step
                sess.run(init_eta_ph, feed_dict={eta_ph: old_eta})
                for j in range(num_e_steps):
                    new_eta, _, _, _, _, _, acc, _ = sess.run(
                        sample_op, feed_dict={x: x_batch,
                                              log_alpha: current_log_alpha})
                    accs.append(acc)
                    # Store eta for the persistent chain
                    if j + 1 == num_e_steps:
                        Eta[t * D:(t + 1) * D, :] = new_eta[0]

                # M step
                _, ll = sess.run([infer, log_likelihood],
                                 feed_dict={x: x_batch,
                                            learning_rate_ph: learning_rate,
                                            log_alpha: current_log_alpha})
                lls.append(ll)

            # MH step for alpha
            current_lp = sess.run(log_hyper_post,
                                  feed_dict={log_alpha: current_log_alpha})
            for m in range(num_mh):
                new_log_alpha = current_log_alpha + np.random.randn() * \
                    mh_stdev
                new_lp = sess.run(log_hyper_post,
                                  feed_dict={log_alpha: new_log_alpha})
                if np.random.rand() < np.exp(min(new_lp - current_lp, 0)):
                    current_lp = new_lp
                    current_log_alpha = new_log_alpha

            time_epoch += time.time()

            print('Epoch {} ({:.1f}s): Perplexity = {}, acc = {}, '
                  'alpha = {}'.format(
                epoch, time_epoch, np.exp(-np.sum(lls) / T), np.mean(accs),
                np.exp(current_log_alpha)))

        # Output topics
        p = sess.run(phi)
        for k in range(K):
            rank = zip(list(p[k, :]), range(V))
            rank.sort()
            rank.reverse()
            sys.stdout.write('Topic {}: '.format(k))
            for i in range(10):
                sys.stdout.write(vocab[rank[i][1]] + ' ')
            sys.stdout.write('\n')
