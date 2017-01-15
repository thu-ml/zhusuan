#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import sys
import os
import time

import tensorflow as tf
from six.moves import range
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from zhusuan.model import *
    from zhusuan.variational import advi
except:
    raise ImportError()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import dataset
except:
    raise ImportError()


class BayesianNN:
    """
    A Bayesian neural network.

    :param x: A Tensor. The data, or mini-batch of data.
    :param layer_sizes: A list of Int. The dimensions of all layers.
    :param n_particles: A Tensor or int. The number of particles per node.
    :param N: Int. The total number of training data.
    """
    def __init__(self, x, layer_sizes, n_particles, N):
        self.N = N

        with StochasticGraph() as model:
            ws = []
            for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
                w_mu = tf.zeros([n_particles, n_out, n_in + 1])
                w_logstd = tf.zeros([n_particles, n_out, n_in + 1])
                ws.append(Normal(w_mu, w_logstd))
            ly_x = self._forward([w.value for w in ws], x)
            y_logstd = tf.Variable(0.)
            y = Normal(ly_x, y_logstd * tf.ones_like(ly_x))

        self.model = model
        self.ws = ws
        self.ly_x = ly_x
        self.y = y
        self.y_logstd = y_logstd

        self.n_particles = n_particles
        self.layer_sizes = layer_sizes

    def log_prob(self, latent, observed, given):
        """
        The log joint probability function.

        :param latent: A dictionary of pairs: (string, Tensor). Each of the
            Tensor has shape (n_particles, ...).
        :param observed: A dictionary of pairs: (string, Tensor). Each of
            the Tensor has shape (n, n_observed).

        :return: A Tensor of shape (n_particles,). The joint log likelihoods.
        """
        y = tf.squeeze(observed['y'], [1])
        y = tf.tile(tf.expand_dims(y, 0), [self.n_particles, 1])
        ws = [latent['w' + str(i)] for i in range(len(self.layer_sizes) - 1)]

        w_dict = dict(zip(self.ws, ws))
        inputs = {self.y: y}
        inputs.update(w_dict)
        out = self.model.get_output([self.y] + self.ws, inputs=inputs)
        y_out = out[0]
        w_outs = out[1:]

        log_py_xw = tf.reduce_mean(y_out[1], 1) * self.N
        log_pw = sum([tf.reduce_sum(w_logp, [-1, -2]) for _, w_logp in w_outs])
        return log_py_xw + log_pw

    def _forward(self, ws, x):
        """
        Get the network output of x with latent variables.
        :param ws: Tensors of weights that has shape (n_particles, ...).
        :param x: A Tensor of shape (n, n_x)

        :return: A Tensor of shape (n_particles, n)
        """
        ws = [tf.tile(tf.expand_dims(w, 1), [1, tf.shape(x)[0], 1, 1])
              for w in ws]
        x = tf.tile(tf.expand_dims(tf.expand_dims(x, 2), 0),
                    [tf.shape(ws[0])[0], 1, 1, 1])
        for i in range(len(ws)):
            x = tf.concat_v2(
                [x, tf.ones([tf.shape(x)[0], tf.shape(x)[1], 1, 1])], 2)
            x = tf.matmul(ws[i], x) / \
                tf.sqrt(tf.cast(tf.shape(x)[2], tf.float32))
            if i < len(ws) - 1:
                x = tf.nn.relu(x)

        return tf.squeeze(x, [2, 3])

    def evaluate(self, latent, observed, std_y_train):
        """
        Calculate the rmse and log likelihood.
        """
        y = tf.squeeze(observed['y'], [1])
        ws = [latent['w' + str(i)] for i in range(len(self.layer_sizes) - 1)]

        inputs = dict(zip(self.ws, ws))
        y_out, _ = self.model.get_output(self.ly_x, inputs=inputs)
        y_pred = tf.reduce_mean(y_out, 0)
        rmse = tf.sqrt(tf.reduce_mean((y_pred - y)**2)) * std_y_train

        mean = tf.reduce_mean(y_out, 0, keep_dims=True)
        variance = tf.reduce_mean((y_out - mean)**2, 0)
        variance = variance + tf.exp(self.y_logstd)**2
        ll = tf.reduce_mean(-0.5 *
                            tf.log(2 * np.pi * variance * std_y_train**2) -
                            0.5 * (y - y_pred)**2 / variance)
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

    x_train, x_test, _, _ = dataset.standardize(x_train, x_test)
    y_train, y_test, mean_y_train, std_y_train = dataset.standardize(
        y_train, y_test)
    std_y_train = np.squeeze(std_y_train)

    # Define training/evaluation parameters
    lb_samples = 10
    ll_samples = 5000
    epoches = 500
    batch_size = 10
    iters = int(np.floor(x_train.shape[0] / float(batch_size)))
    test_freq = 10
    learning_rate = 0.01
    anneal_lr_freq = 100
    anneal_lr_rate = 1.

    # Build training model
    learning_rate_ph = tf.placeholder(tf.float32, shape=())
    n_particles = tf.placeholder(tf.int32, shape=[], name='n_particles')
    x = tf.placeholder(tf.float32, shape=(None, n_x))
    layer_sizes = [n_x, 50, 1]
    bnn = BayesianNN(x, layer_sizes, n_particles, x_train.shape[0])
    y = tf.placeholder(tf.float32, shape=(None, 1))
    observed = {'y': y}
    optimizer = tf.train.AdamOptimizer(learning_rate_ph, epsilon=1e-4)

    # Build variational posterior
    with StochasticGraph() as variational:
        w = []
        for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            mu = tf.Variable(tf.zeros([n_out, n_in + 1]))
            logstd = tf.Variable(tf.zeros([n_out, n_in + 1]))
            w.append(Normal(mu, logstd, sample_dim=0, n_samples=n_particles))

    w_outputs = variational.get_output(w)
    w_outputs = [[qw, tf.reduce_sum(qw_logpdf, [-1, -2])]
                 for qw, qw_logpdf in w_outputs]
    latent = {'w' + str(i): w_outputs[i] for i in range(len(w_outputs))}
    lower_bound = tf.reduce_mean(advi(
        bnn, observed, latent, reduction_indices=0))
    grads = optimizer.compute_gradients(-lower_bound)
    infer = optimizer.apply_gradients(grads)

    latent_outputs = {'w' + str(i): w_outputs[i][0]
                      for i in range(len(w_outputs))}
    rmse, ll = bnn.evaluate(latent_outputs, observed, std_y_train)

    params = tf.trainable_variables()
    for i in params:
        print(i.name, i.get_shape())

    # Run the inference
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(1, epoches + 1):
            time_epoch = -time.time()
            if epoch % anneal_lr_freq == 0:
                learning_rate *= anneal_lr_rate
            lbs = []
            for t in range(iters):
                x_batch = x_train[t * batch_size:(t + 1) * batch_size]
                y_batch = y_train[t * batch_size:(t + 1) * batch_size]
                _, lb = sess.run(
                    [infer, lower_bound],
                    feed_dict={n_particles: lb_samples,
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
                    feed_dict={n_particles: ll_samples,
                               x: x_test, y: y_test})
                time_test += time.time()
                print('>>> TEST ({:.1f}s)'.format(time_test))
                print('>> Test lower bound = {}'.format(test_lb))
                print('>> Test rmse = {}'.format(test_rmse))
                print('>> Test log_likelihood = {}'.format(test_ll))
