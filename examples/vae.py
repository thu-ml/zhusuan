#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import sys
import os
import time

import tensorflow as tf
import prettytensor as pt
from six.moves import range
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from zhusuan.distributions import norm, bernoulli
    from zhusuan.layers import *
    from zhusuan.variational import advi
    from zhusuan.evaluation import is_loglikelihood
except:
    raise ImportError()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import dataset
except:
    raise ImportError()


class M1:
    """
    The deep generative model used in variational autoencoder (VAE).

    :param n_z: Int. The dimension of latent variables (z).
    :param n_x: Int. The dimension of observed variables (x).
    """
    def __init__(self, n_z, n_x):
        self.n_z = n_z
        self.n_x = n_x
        with pt.defaults_scope(activation_fn=tf.nn.relu):
            self.l_x_z = (pt.template('z').
                          fully_connected(500).
                          batch_normalize(scale_after_normalization=True).
                          fully_connected(500).
                          batch_normalize(scale_after_normalization=True).
                          fully_connected(n_x, activation_fn=tf.nn.sigmoid))

    def log_prob(self, latent, observed):
        """
        The log joint probability function.

        :param latent: A dictionary of pairs: (string, Tensor). Each of the
            Tensor has shape (batch_size, n_samples, n_latent).
        :param observed: A dictionary of pairs: (string, Tensor). Each of the
            Tensor has shape (batch_size, n_observed).

        :return: A Tensor of shape (batch_size, n_samples). The joint log
            likelihoods.
        """
        z = latent['z']
        x = observed['x']

        l_x_z = self.l_x_z.construct(
            z=tf.reshape(z, (-1, self.n_z))).reshape(
            (-1, tf.shape(z)[1], self.n_x)).tensor
        log_px_z = tf.reduce_sum(
            bernoulli.logpdf(tf.expand_dims(x, 1), l_x_z, eps=1e-6), 2)
        log_pz = tf.reduce_sum(norm.logpdf(z), 2)
        return log_px_z + log_pz


def q_net(n_x, n_z, n_samples):
    """
    Build the recognition network (Q-net) used as variational posterior.

    :param n_x: Int. The dimension of observed variables (x).
    :param n_z: Int. The dimension of latent variables (z).
    :param n_samples: A Int or a Tensor of type int. Number of samples of
        latent variables.

    :return: All :class:`Layer` instances needed.
    """
    with pt.defaults_scope(activation_fn=tf.nn.relu):
        lx = InputLayer((None, n_x))
        lz_x = PrettyTensor({'x': lx}, pt.template('x').
                            fully_connected(500).
                            batch_normalize(scale_after_normalization=True).
                            fully_connected(500).
                            batch_normalize(scale_after_normalization=True))
        lz_mean = PrettyTensor({'z': lz_x}, pt.template('z').
                               fully_connected(n_z, activation_fn=None).
                               reshape((-1, 1, n_z)))
        lz_logstd = PrettyTensor({'z': lz_x}, pt.template('z').
                                 fully_connected(n_z, activation_fn=None).
                                 reshape((-1, 1, n_z)))
        lz = ReparameterizedNormal([lz_mean, lz_logstd], n_samples)
    return lx, lz


if __name__ == "__main__":
    tf.set_random_seed(1237)

    # Load MNIST
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'data', 'mnist.pkl.gz')
    x_train, t_train, x_valid, t_valid, x_test, t_test = \
        dataset.load_mnist_realval(data_path)
    x_train = np.vstack([x_train, x_valid]).astype('float32')
    np.random.seed(1234)
    x_test = np.random.binomial(1, x_test, size=x_test.shape).astype('float32')
    n_x = x_train.shape[1]

    # Define model parameters
    n_z = 40

    # Define training/evaluation parameters
    lb_samples = 1
    ll_samples = 5000
    epoches = 3000
    batch_size = 100
    test_batch_size = 100
    iters = x_train.shape[0] // batch_size
    test_iters = x_test.shape[0] // test_batch_size
    test_freq = 10
    learning_rate = 0.001
    anneal_lr_freq = 200
    anneal_lr_rate = 0.75

    # Build the training computation graph
    learning_rate_ph = tf.placeholder(tf.float32, shape=[])
    x = tf.placeholder(tf.float32, shape=(None, x_train.shape[1]))
    n_samples = tf.placeholder(tf.int32, shape=())
    optimizer = tf.train.AdamOptimizer(learning_rate_ph, epsilon=1e-4)
    with pt.defaults_scope(phase=pt.Phase.train):
        with tf.variable_scope("model") as scope:
            train_model = M1(n_z, x_train.shape[1])
        with tf.variable_scope("variational") as scope:
            lx, lz = q_net(n_x, n_z, n_samples)
    grads, lower_bound = advi(
        train_model, {'x': x}, {'x': lx}, {'z': lz}, optimizer)
    infer = optimizer.apply_gradients(grads)

    # Build the evaluation computation graph
    with pt.defaults_scope(phase=pt.Phase.test):
        with tf.variable_scope("model", reuse=True) as scope:
            eval_model = M1(n_z, x_train.shape[1])
        with tf.variable_scope("variational", reuse=True) as scope:
            lx, lz = q_net(n_x, n_z, n_samples)
    _, eval_lower_bound = advi(
        eval_model, {'x': x}, {'x': lx}, {'z': lz}, optimizer)
    eval_log_likelihood = is_loglikelihood(
        eval_model, {'x': x}, {'x': lx}, {'z': lz})

    params = tf.trainable_variables()
    for i in params:
        print(i.name, i.get_shape())

    init = tf.initialize_all_variables()

    # Run the inference
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(1, epoches + 1):
            time_epoch = -time.time()
            if epoch % anneal_lr_freq == 0:
                learning_rate *= anneal_lr_rate
            np.random.shuffle(x_train)
            lbs = []
            for t in range(iters):
                x_batch = x_train[t * batch_size:(t + 1) * batch_size]
                x_batch = np.random.binomial(
                    n=1, p=x_batch, size=x_batch.shape).astype('float32')
                _, lb = sess.run([infer, lower_bound],
                                 feed_dict={x: x_batch,
                                            learning_rate_ph: learning_rate,
                                            n_samples: lb_samples})
                lbs.append(lb)
            time_epoch += time.time()
            print('Epoch {} ({:.1f}s): Lower bound = {}'.format(
                epoch, time_epoch, np.mean(lbs)))
            if epoch % test_freq == 0:
                time_test = -time.time()
                test_lbs = []
                test_lls = []
                for t in range(test_iters):
                    test_x_batch = x_test[
                        t * test_batch_size: (t + 1) * test_batch_size]
                    test_lb = sess.run(eval_lower_bound,
                                       feed_dict={x: test_x_batch,
                                                  n_samples: lb_samples})
                    test_ll = sess.run(eval_log_likelihood,
                                       feed_dict={x: test_x_batch,
                                                  n_samples: ll_samples})
                    test_lbs.append(test_lb)
                    test_lls.append(test_ll)
                time_test += time.time()
                print('>>> TEST ({:.1f}s)'.format(time_test))
                print('>> Test lower bound = {}'.format(np.mean(test_lbs)))
                print('>> Test log likelihood = {}'.format(np.mean(test_lls)))
