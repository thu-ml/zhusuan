#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Logistic-normal topic models using Monte-Carlo EM
Dense implementation, O(n_docs*n_topics*n_vocab)
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import sys
import os
import time

import tensorflow as tf
from six.moves import range, zip
from functools import partial
import numpy as np
import zhusuan as zs

from examples import conf
from examples.utils import dataset


# Delta in LNTM is corresponding to eta in LDA(Blei et al., 2003),
# which governs the prior of parameter in topic->word categorical distribution.
# Larger log_delta leads to sparser topics.
log_delta = 10.0


def lntm(observed, n_chains, n_docs, n_topics, n_vocab, eta_mean, eta_logstd):
    with zs.BayesianNet(observed=observed) as model:
        eta_mean = tf.tile(tf.expand_dims(eta_mean, 0), [n_docs, 1])
        # eta/theta: Unnormalized/normalized document-topic matrix
        eta = zs.Normal('eta', eta_mean, logstd=eta_logstd, n_samples=n_chains,
                        group_ndims=1)
        theta = tf.nn.softmax(eta)
        # beta/phi: Unnormalized/normalized topic-word matrix
        beta = zs.Normal('beta', tf.zeros([n_topics, n_vocab]),
                         logstd=log_delta, group_ndims=1)
        phi = tf.nn.softmax(beta)
        # doc_word: Document-word matrix
        doc_word = tf.matmul(tf.reshape(theta, [-1, n_topics]), phi)
        doc_word = tf.reshape(doc_word, [n_chains, n_docs, n_vocab])
        x = zs.UnnormalizedMultinomial('x', tf.log(doc_word),
                                       normalize_logits=False,
                                       dtype=tf.float32)
    return model


if __name__ == "__main__":
    tf.set_random_seed(1237)

    # Load nips dataset
    data_name = 'nips'
    data_path = os.path.join(conf.data_dir, data_name + '.pkl.gz')
    X, vocab = dataset.load_uci_bow(data_name, data_path)
    training_size = 1200
    X_train = X[:training_size, :]
    X_test = X[training_size:, :]

    # Define model training parameters
    batch_size = 100
    n_topics = 100
    n_vocab = X_train.shape[1]
    n_chains = 1

    num_e_steps = 5
    hmc = zs.HMC(step_size=1e-3, n_leapfrogs=20, adapt_step_size=True,
                 target_acceptance_rate=0.6)
    epochs = 100
    learning_rate_0 = 1.0
    t0 = 10

    # Padding
    rem = batch_size - X_train.shape[0] % batch_size
    if rem < batch_size:
        X_train = np.vstack((X_train, np.zeros((rem, n_vocab))))

    iters = X_train.shape[0] // batch_size
    Eta = np.zeros((n_chains, X_train.shape[0], n_topics), dtype=np.float32)
    Eta_mean = np.zeros(n_topics, dtype=np.float32)
    Eta_logstd = np.zeros(n_topics, dtype=np.float32)

    # Build the computation graph
    x = tf.placeholder(tf.float32, shape=[batch_size, n_vocab], name='x')
    eta_mean = tf.placeholder(tf.float32, shape=[n_topics], name='eta_mean')
    eta_logstd = tf.placeholder(tf.float32, shape=[n_topics],
                                name='eta_logstd')
    eta = tf.Variable(tf.zeros([n_chains, batch_size, n_topics]), name='eta')
    eta_ph = tf.placeholder(tf.float32, shape=[n_chains, batch_size, n_topics],
                            name='eta_ph')
    beta = tf.Variable(tf.zeros([n_topics, n_vocab]), name='beta')
    phi = tf.nn.softmax(beta)
    init_eta_ph = tf.assign(eta, eta_ph)

    def e_obj(observed, n_chains, n_docs):
        model = lntm(observed, n_chains, n_docs, n_topics, n_vocab,
                     eta_mean, eta_logstd)
        return model.local_log_prob('eta') + model.local_log_prob('x')

    # E step: sample eta using HMC
    sample_op, hmc_info = hmc.sample(partial(e_obj, n_chains=n_chains,
                                             n_docs=batch_size),
                                     observed={'x': x, 'beta': beta},
                                     latent={'eta': eta})
    # M step: optimize beta
    model = lntm({'x': x, 'eta': eta, 'beta': beta}, n_chains, batch_size,
                 n_topics, n_vocab, eta_mean, eta_logstd)
    log_p_beta, log_px = model.local_log_prob(['beta', 'x'])
    log_likelihood = tf.reduce_sum(tf.reduce_mean(log_px, axis=0))
    log_joint = tf.reduce_sum(log_p_beta) + log_likelihood
    learning_rate_ph = tf.placeholder(tf.float32, shape=[], name='lr')
    optimizer = tf.train.AdamOptimizer(learning_rate_ph)
    infer = optimizer.minimize(-log_joint, var_list=[beta])

    params = tf.trainable_variables()
    for i in params:
        print(i.name, i.get_shape())

    # Below is the evaluation part.
    # Variables whose name starts with '_' is only used in the evaluation part,
    # to be distinguished from those variables used in the training part above.

    n_docs_test = X_test.shape[0]
    _n_chains = 25
    _n_temperatures = 1000

    _x = tf.placeholder(tf.float32, shape=[n_docs_test, n_vocab], name='x')
    _eta = tf.Variable(tf.zeros([_n_chains, n_docs_test, n_topics]),
                       name='eta')

    def _log_prior(observed):
        return lntm(observed, _n_chains, n_docs_test, n_topics, n_vocab,
                    eta_mean, eta_logstd).local_log_prob('eta')

    _prior_samples = {'eta': lntm({}, _n_chains, n_docs_test, n_topics,
                      n_vocab, eta_mean, eta_logstd).outputs('eta')}

    _hmc = zs.HMC(step_size=0.01, n_leapfrogs=20, adapt_step_size=True,
                  target_acceptance_rate=0.6)

    _ais = zs.evaluation.AIS(_log_prior,
                             partial(e_obj, n_chains=_n_chains,
                                     n_docs=n_docs_test),
                             _prior_samples, _hmc,
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
                x_batch = X_train[t*batch_size: (t+1)*batch_size]
                old_eta = Eta[:, t*batch_size: (t+1)*batch_size, :]

                # E step
                sess.run(init_eta_ph, feed_dict={eta_ph: old_eta})
                for j in range(num_e_steps):
                    _, new_eta, acc = sess.run(
                        [sample_op, hmc_info.samples['eta'],
                         hmc_info.acceptance_rate],
                        feed_dict={x: x_batch,
                                   eta_mean: Eta_mean,
                                   eta_logstd: Eta_logstd})
                    accs.append(acc)
                    # Store eta for the persistent chain
                    if j + 1 == num_e_steps:
                        Eta[:, t*batch_size: (t+1)*batch_size, :] = new_eta

                # M step
                _, ll = sess.run(
                    [infer, log_likelihood],
                    feed_dict={x: x_batch,
                               eta_mean: Eta_mean,
                               eta_logstd: Eta_logstd,
                               learning_rate_ph: learning_rate * t0 / (
                                   t0 + epoch)})
                lls.append(ll)

            # Update hyper-parameters
            Eta_mean = np.mean(Eta, axis=(0, 1))
            Eta_logstd = np.log(np.std(Eta, axis=(0, 1)) + 1e-6)

            time_epoch += time.time()
            print('Epoch {} ({:.1f}s): Perplexity = {:.2f}, acc = {:.3f}, '
                  'eta mean = {:.2f}, logstd = {:.2f}'
                  .format(epoch, time_epoch,
                          np.exp(-np.sum(lls) / np.sum(X_train)),
                          np.mean(accs), np.mean(Eta_mean),
                          np.mean(Eta_logstd)))

        # Output topics
        p = sess.run(phi)
        for k in range(n_topics):
            rank = list(zip(list(p[k, :]), range(n_vocab)))
            rank.sort()
            rank.reverse()
            sys.stdout.write('Topic {}, eta mean = {:.2f} stdev = {:.2f}: '
                             .format(k, Eta_mean[k], np.exp(Eta_logstd[k])))
            for i in range(10):
                sys.stdout.write(vocab[rank[i][1]] + ' ')
            sys.stdout.write('\n')

        # Run AIS
        print("Evaluating test perplexity using AIS...")

        time_ais = -time.time()

        ll_lb = _ais.run(sess, feed_dict={_x: X_test,
                                          eta_mean: Eta_mean,
                                          eta_logstd: Eta_logstd})

        time_ais += time.time()

        print('>> Test (AIS) ({:.1f}s)\n'
              '>> log likelihood lower bound = {}\n'
              '>> perplexity upper bound = {}'
              .format(time_ais, ll_lb,
                      np.exp(-ll_lb * n_docs_test / np.sum(X_test))))
