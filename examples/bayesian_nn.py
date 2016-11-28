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
from dataset import standardize

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


class BayesianNN():
    """
    A Bayesian neural network.

    :param layer_sizes: A list of Int. The dimensions of all layers.
    :param n: A Tensor or int. The number of data, or batch size in mini-batch
        training.
    :param n_particles: A Tensor or int. The number of particles per node.
    :param y_logstd: Float. Given log standard deviation of q(y|w, x).
    :param N: Int. Number of training data.
    """
    def __init__(self, layer_sizes, n, n_particles, y_logstd, N):
        self.y_logstd = y_logstd
        self.N = N

        with StochasticGraph() as model:
            #define parameters
            def sample_param(layer_sizes):
                w = []
                for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
                    mu = tf.zeros([n_particles, n_out, n_in + 1])
                    logstd = tf.zeros([n_particles, n_out, n_in + 1])
                    w.append(Normal(mu, logstd))
                return w
            w = sample_param(layer_sizes)

            #define x
            x = tf.placeholder(tf.float32, [None, layer_sizes[0]])

            y = self._forward([item.value for item in w], x)
            y_sample = Normal(y, self.y_logstd * tf.ones_like(y))

        self.model = model
        self.w = w
        self.x = x
        self.y = y
        self.y_sample = y_sample

        self.n_particles = n_particles
        self.layer_sizes = layer_sizes

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
        y = tf.squeeze(observed['y'])
        y = tf.tile(tf.expand_dims(y, 1), [1, self.n_particles])
        w = [latent['w'+str(i)] for i in range(len(self.layer_sizes)-1)]

        w_dict = {self.w[i]: w[i] for i in range(len(w))}
        inputs = {self.x: x, self.y_sample: y}
        inputs.update(w_dict)
        out = self.model.get_output([self.y_sample] + self.w,
                                    inputs=inputs)
        y_out = out[0]
        w_out = out[1:]

        log_py_x = tf.reduce_mean(y_out[1], 0) * self.N
        log_pw = sum([tf.reduce_sum(item[1], [-1, -2]) for item in w_out])
        return log_py_x + log_pw

    def _forward(self, w, x):
        """
        get the network output of x with latent variables.
        :param w1: A Tensor has shape (n_samples, ...).
        :param w2: A Tensor has shape (n_samples, ...).
        :param x: A Tensor has shape (batch_size, n_observed_x)
        :param given: A dict.

        :return: A Tensor of shape (batch_size, n_samples)
        """
        w = [tf.tile(tf.expand_dims(item, 0), [tf.shape(x)[0], 1, 1, 1])
             for item in w]

        x = tf.tile(tf.expand_dims(tf.expand_dims(x, 2), 1),
                    [1, tf.shape(w[0])[1], 1, 1])
        for i in range(len(w)):
            x = tf.concat(2, [x, tf.ones((tf.shape(x)[0],
                                          tf.shape(x)[1], 1, 1))])
            x = tf.batch_matmul(w[i], x) / \
                tf.sqrt(tf.cast(tf.shape(x)[2], tf.float32))

            if i < len(w)-1:
                x = tf.nn.relu(x)

        return tf.squeeze(x, [2, 3])

    def evaluate(self, latent, observed, std_y_train=1.):
        """
        Calculate the rmse and log likelihood.
        """
        x = observed['x']
        y = tf.squeeze(observed['y'], 1)
        w = [latent['w'+str(i)] for i in range(len(self.layer_sizes)-1)]

        w_dict = {self.w[i]: w[i] for i in range(len(w))}
        inputs = {self.x: x}
        inputs.update(w_dict)
        y_out = self.model.get_output(self.y, inputs=inputs)[0]
        y_out_mean = tf.reduce_mean(y_out, 1)

        rmse = tf.sqrt(tf.reduce_mean((y_out_mean - y)**2))\
            * std_y_train

        mean = tf.tile(tf.expand_dims(y_out_mean, 1),
                       [1, tf.shape(y_out)[1]])
        variance = tf.reduce_mean((y_out - mean)**2, 1)
        variance = variance + np.exp(self.y_logstd)
        ll = tf.reduce_mean(-0.5 *
                            tf.log(2 * np.pi * variance * std_y_train**2) -
                            0.5 * (y - y_out_mean)**2 / variance)
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
    y_logstd = np.log(0.1)

    # Define training/evaluation parameters
    lb_samples = 10
    ll_samples = 500
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
    x = tf.placeholder(tf.float32, shape=(None, x_train.shape[1]))
    n = tf.shape(x)[0]
    layer_sizes = [x_train.shape[1], 50, 1]
    model = BayesianNN(layer_sizes, n, n_particles, y_logstd, x_train.shape[0])
    y = tf.placeholder(tf.float32, shape=(None, 1))
    observed = {'x': x, 'y': y}

    optimizer = tf.train.AdamOptimizer(learning_rate_ph, epsilon=1e-4)

    # Build variational posterior
    with StochasticGraph() as variational:
        w = []
        for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            mu = tf.Variable(tf.zeros([n_out, n_in + 1]))
            logstd = tf.Variable(tf.zeros([n_out, n_in + 1]))
            w.append(Normal(mu, logstd, sample_dim=0, n_samples=n_particles))

    w_outputs = variational.get_output(w)
    w_outputs = [[item[0], tf.reduce_sum(item[1], [-1, -2])]
                 for item in w_outputs]
    latent = {'w'+str(i): w_outputs[i] for i in range(len(w_outputs))}

    lower_bound = -tf.reduce_mean(advi(
        model, observed, latent, reduction_indices=0))
    grads = optimizer.compute_gradients(lower_bound)
    infer = optimizer.apply_gradients(grads)

    latent_outputs = {'w'+str(i): w_outputs[i][0]
                      for i in range(len(w_outputs))}
    rmse, ll = model.evaluate(latent_outputs, observed,
                              std_y_train=std_y_train)

    params = tf.trainable_variables()
    for i in params:
        print(i.name, i.get_shape())

    # Run the inference
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for epoch in range(1, epoches + 1):
            time_epoch = -time.time()
            if epoch % anneal_lr_freq == 0:
                learning_rate *= anneal_lr_rate
            lbs = []
            for t in range(iters):
                x_batch = x_train[t * batch_size:(t + 1) * batch_size]
                y_batch = y_train[t * batch_size:(t + 1) * batch_size]

                _, _, lb = sess.run(
                    [infer, grads, lower_bound],
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
