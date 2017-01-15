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
from dataset import standardize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from zhusuan.distributions_old import norm, bernoulli
    from zhusuan.layers_old import *
    from zhusuan.variational import advi
except:
    raise ImportError()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import dataset
except:
    raise ImportError()


class BayesianNN():
    """
    A Bayesian neural network.

    :param scale: Float. Given standard deviation of q(y|w, x).
    :param N: Int. Number of training data.
    """
    def __init__(self, scale, N):
        self.scale = scale
        self.N = N

    def log_prob(self, latent, observed, given):
        """
        The log joint probability function.

        :param latent: A dictionary of pairs: (string, Tensor). Each of the
            Tensor has shape (1, n_samples, ...).
        :param observed: A dictionary of pairs: (string, Tensor). Each of
            the Tensor has shape (batch_size, n_observed).

        :return: A Tensor of shape (batch_size, n_samples). The joint log
            likelihoods.
        """
        x = observed['x']
        y = observed['y']

        network_output = self._forward(latent, x)
        network_output = tf.squeeze(network_output, [2, 3])
        y = tf.tile(tf.expand_dims(y, 1), [1, tf.shape(network_output)[1], 1])
        y = tf.squeeze(y, [2])
        log_likelihood = norm.logpdf(y, network_output, self.scale)
        log_likelihood = tf.reduce_mean(log_likelihood, 0) * self.N
        log_likelihood = tf.expand_dims(log_likelihood, 0)
        log_prior = sum([tf.reduce_sum(norm.logpdf(item))
                        for item in latent.values()])
        return log_likelihood + log_prior

    def _forward(self, latent, x):
        """
        get the network output of x with latent variables.
        :param latent: A dictionary of pairs: (string, Tensor). Each of the
            Tensor has shape (1, n_samples, shape_latent).
        :param x: A dictionary of pairs: (string, Tensor). Each of the
            Tensor has shape (batch_size, n_observed_x)

        :return: A Tensor of shape (batch_size, n_samples, n_observed_y)
        """
        w1 = latent['w1']
        w2 = latent['w2']

        w1 = tf.tile(w1, [tf.shape(x)[0], 1, 1, 1])
        w2 = tf.tile(w2, [tf.shape(x)[0], 1, 1, 1])

        x = tf.tile(tf.expand_dims(tf.expand_dims(x, 2), 1),
                    [1, tf.shape(w1)[1], 1, 1])
        x = tf.concat_v2([x, tf.ones((tf.shape(x)[0], tf.shape(x)[1], 1, 1))],
                         2)

        l = tf.matmul(w1, x) / \
            tf.sqrt(tf.cast(tf.shape(x)[2], tf.float32))
        l = tf.concat_v2([l, tf.ones((tf.shape(l)[0], tf.shape(l)[1], 1, 1))],
                         2)
        l = tf.nn.relu(l)

        y = tf.matmul(w2, l) / \
            tf.sqrt(tf.cast(tf.shape(l)[2], tf.float32))
        return y

    def evaluate(self, latent, observed, std_y_train=1.):
        """
        Calculate the rmse and log likelihood.
        """
        network_output = self._forward(latent, observed['x'])
        network_output = tf.squeeze(network_output, [2, 3])
        network_output_mean = tf.reduce_mean(network_output, 1)

        y = observed['y']
        y = tf.squeeze(y, [1])

        rmse = tf.sqrt(tf.reduce_mean((network_output_mean - y)**2))\
            * std_y_train

        mean = tf.tile(tf.expand_dims(network_output_mean, 1),
                       [1, tf.shape(network_output)[1]])
        variance = tf.reduce_mean((network_output - mean)**2, 1)
        variance = variance + self.scale
        ll = tf.reduce_mean(-0.5 *
                            tf.log(2 * np.pi * variance * std_y_train**2) -
                            0.5 * (y - network_output_mean)**2 / variance)
        return rmse, ll


if __name__ == '__main__':
    tf.set_random_seed(1237)
    np.random.seed(1234)

    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'data', 'housing.data')
    x_train, y_train, x_valid, y_valid, x_test, y_test = \
        dataset.load_uci_boston_housing(data_path)
    x_train = np.vstack([x_train, x_valid]).astype('float32')
    y_train = np.hstack((y_train, y_valid)).astype('float32')
    y_train = y_train.reshape((len(y_train), 1))

    x_test = x_test.astype('float32')
    y_test = y_test.reshape((len(y_test), 1))
    n_x = x_train.shape[1]

    x_train, x_test, _, _ = standardize(x_train, x_test)
    y_train, y_test, mean_y_train, std_y_train = standardize(y_train, y_test)
    std_y_train = np.squeeze(std_y_train)

    # Define model parameters
    y_std = 0.1

    # Define training/evaluation parameters
    lb_samples = 10
    ll_samples = 500
    epoches = 500
    batch_size = 10
    iters = int(np.floor(x_train.shape[0] / float(batch_size)))
    test_freq = 10
    learning_rate = 0.01
    anneal_lr_freq = 100
    anneal_lr_rate = 0.75

    # Build training model
    model = BayesianNN(y_std, x_train.shape[0])
    learning_rate_ph = tf.placeholder(tf.float32, shape=())
    x = tf.placeholder(tf.float32, shape=(None, x_train.shape[1]))
    y = tf.placeholder(tf.float32, shape=(None, 1))
    observed = {'x': x, 'y': y}
    n_samples = tf.placeholder(tf.int32, shape=())
    optimizer = tf.train.AdamOptimizer(learning_rate_ph, epsilon=1e-4)

    # shape: (batch_size, n_samples, ...)
    w1_mu = tf.Variable(tf.zeros([1, 1, 50, x_train.shape[1] + 1]))
    w1_logvar = tf.Variable(tf.zeros([1, 1, 50, x_train.shape[1] + 1]))
    w2_mu = tf.Variable(tf.zeros([1, 1, 1, 50 + 1]))
    w2_logvar = tf.Variable(tf.zeros([1, 1, 1, 50 + 1]))

    # Build variational posterior
    lw1_mu = InputLayer(w1_mu.get_shape().as_list(), input=w1_mu)
    lw1_logvar = InputLayer(w1_logvar.get_shape().as_list(), input=w1_logvar)
    lw2_mu = InputLayer(w2_mu.get_shape().as_list(), input=w2_mu)
    lw2_logvar = InputLayer(w2_logvar.get_shape().as_list(), input=w2_logvar)
    lw1 = Normal([lw1_mu, lw1_logvar], n_samples)
    lw2 = Normal([lw2_mu, lw2_logvar], n_samples)
    w1_outputs, w2_outputs = get_output([lw1, lw2])
    latent = {'w1': w1_outputs, 'w2': w2_outputs}

    lower_bound = tf.reduce_mean(advi(
        model, observed, latent, reduction_indices=1))
    grads = optimizer.compute_gradients(-lower_bound)
    infer = optimizer.apply_gradients(grads)

    latent_outputs = {'w1': w1_outputs[0], 'w2': w2_outputs[0]}
    rmse, ll = model.evaluate(latent_outputs, observed, std_y_train)
    output = tf.reduce_mean(
        model._forward(latent_outputs, x), 1)

    params = tf.trainable_variables()
    for i in params:
        print(i.name, i.get_shape())

    init = tf.global_variables_initializer()

    # Run the inference
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(1, epoches + 1):
            time_epoch = -time.time()
            if epoch % anneal_lr_freq == 0:
                learning_rate *= anneal_lr_rate
            lbs = []
            for t in range(iters):
                x_batch = x_train[t * batch_size:(t + 1) * batch_size]
                y_batch = y_train[t * batch_size:(t + 1) * batch_size]

                _, grad1, lb = sess.run(
                    [infer, grads, lower_bound],
                    feed_dict={n_samples: lb_samples,
                               learning_rate_ph: learning_rate,
                               x: x_batch, y: y_batch})

                lbs.append(lb)
            time_epoch += time.time()
            print('Epoch {} ({:.1f}s): Lower bound = {}'.format(
                epoch, time_epoch, np.mean(lbs)))

            if epoch % test_freq == 0:
                time_test = -time.time()

                test_lb, test_rmse, test_ll = sess.run(
                    [lower_bound, rmse, ll],
                    feed_dict={n_samples: ll_samples,
                               x: x_test, y: y_test})
                time_test += time.time()
                print('>>> TEST ({:.1f}s)'.format(time_test))
                print('>> Test lower bound = {}'.format(test_lb))
                print('>> Test rmse = {}'.format(test_rmse))
                print('>> Test log_likelihood = {}'.format(test_ll))
