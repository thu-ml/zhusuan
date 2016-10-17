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
from dataset import load_uci_boston_housing, standardize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from zhusuan.distributions import norm, bernoulli
    from zhusuan.layers import *
    from zhusuan.variational import advi
except:
    raise ImportError()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import dataset
except:
    raise ImportError()


class BayesianMV_NN():
    """
    A Bayesian neural network with matrix variate posterior.

    :param scale: Float. Given standard deviation of q(y|w, x).
    :param N: Int. Number of training data.
    :layer_Num: Int. Number of layers
    """
    def __init__(self, scale, N, layer_Num):
        self.scale = scale
        self.N = N
        self.layer_Num = layer_Num

    def log_prob(self, latent, observed, given):
        """
        The log joint probability function.
        :param latent: A dictionary of pairs: (string, Tensor). Each of the
            Tensor has shape (1, n_samples, ...).
        :param observed: A dictionary of pairs: (string, Tensor). Each of
            the Tensor has shape (batch_size, n_observed).
        :param given: A dictionary of pairs: (string, Tensor). Each of the
            Tensor has shape (1, 1, n_observed, 1)

        :return: A Tensor of shape (batch_size, n_samples).
            The joint log likelihoods.
        """
        x = observed['x']
        y = observed['y']

        pwq = []
        for i in range(self.layer_Num - 1):
            pwq.append(self._multiply(given['pv'+str(i)],
                                      latent['w'+str(i)],
                                      given['qv'+str(i)]))

        network_output = self._forward(pwq, x)
        network_output = tf.squeeze(network_output, [2, 3])
        y = tf.tile(tf.expand_dims(y, 1), [1, tf.shape(network_output)[1], 1])
        y = tf.squeeze(y, [2])
        log_likelihood = norm.logpdf(y, network_output, self.scale)
        log_likelihood = tf.reduce_mean(log_likelihood, 0) * self.N
        log_likelihood = tf.expand_dims(log_likelihood, 0)

        log_prior = sum([tf.reduce_sum(norm.logpdf(item))
                        for item in latent.values()])
        return log_likelihood + log_prior

    def _multiply(self, pv, w, qv):
        """
        calculate P * W * Q(matrix multiplication).
        with P = I - 2 pv * pv.transpose() / pv.transpose() * pv, so is Q.
        """
        def func(v):
            I = tf.matrix_diag(tf.ones(tf.shape(v)[:3]))
            temp = tf.batch_matmul(v, tf.transpose(v, perm=[0, 1, 3, 2])) / \
                tf.batch_matmul(tf.transpose(v, perm=[0, 1, 3, 2]), v)
            return I - 2 * temp

        P = tf.tile(func(pv), [1, tf.shape(w)[1], 1, 1])
        Q = tf.tile(func(qv), [1, tf.shape(w)[1], 1, 1])
        wq = tf.batch_matmul(w, Q)
        pwq = tf.batch_matmul(P, wq)
        return pwq

    def _forward(self, pwq, x):
        """
        get the network output of x with latent variables.
        :param pwq: A list of tensors: Each of the
            Tensor has shape (1, n_samples, shape_latent).
        :param x: Tensor. Each of the
            Tensor has shape (batch_size, n_observed_x)

        :return: A Tensor of shape (batch_size, n_samples, n_observed_y, 1)
        """

        x = tf.tile(tf.expand_dims(tf.expand_dims(x, 2), 1),
                    [1, tf.shape(pwq[0])[1], 1, 1])

        for i in range(self.layer_Num - 1):
            temp = tf.tile(pwq[i], [tf.shape(x)[0], 1, 1, 1])
            x = tf.concat(2, [x, tf.ones((tf.shape(x)[0],
                                         tf.shape(x)[1], 1, 1))])
            x = tf.batch_matmul(temp, x) / tf.sqrt(tf.cast(tf.shape(x)[2],
                                                           tf.float32))
            if not i == self.layer_Num - 2:
                x = tf.nn.relu(x)

        return x

    def evaluate(self, pwq, observed, std_y_train=1):
        """
        calculate the rmse and log likelihood.
        """
        network_output = self._forward(pwq, observed['x'])
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


def MV_layer(n_in, n_out, n_samples):
    """
    build a layer with matrix variate posterior.
    :param n_in: Int. num of input neurons.
    :param n_out: Int. num of output neurons.
    :param n_samples: Int. Num of samples for Guassian distributed parameters.

    :return: A variable of shape (1, 1, n_in, 1); A Normal;
        A vairable of shape (1, 1, n_out, 1)
    """
    pv = tf.Variable(tf.random_normal([1, 1, n_out, 1]))
    qv = tf.Variable(tf.random_normal([1, 1, n_in, 1]))

    w_mu = tf.Variable(tf.zeros([1, 1, n_out, n_in]))
    w_logvar = tf.Variable(tf.zeros([1, 1, n_out, n_in]))
    w_mu = InputLayer((1, 1, n_out, n_in), input=w_mu)
    w_logvar = InputLayer((1, 1, n_out, n_in), input=w_logvar)
    return qv, Normal([w_mu, w_logvar], n_samples), pv


if __name__ == '__main__':
    tf.set_random_seed(1235)
    np.random.seed(1235)

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

    #build training model
    hidden_units = [50]
    layer_units = [x_train.shape[1]] + hidden_units + [1]
    layer_Num = len(layer_units)

    model = BayesianMV_NN(0.1, x_train.shape[0], layer_Num=layer_Num)

    learning_rate_ph = tf.placeholder(tf.float32, shape=())
    x = tf.placeholder(tf.float32, shape=(None, x_train.shape[1]))
    y = tf.placeholder(tf.float32, shape=(None, 1))
    observed = {'x': x, 'y': y}
    n_samples = tf.placeholder(tf.int32, shape=())
    optimizer = tf.train.AdamOptimizer(learning_rate_ph, epsilon=1e-4)

    latent = {}
    given = {}
    for i in range(layer_Num-1):
        layer = MV_layer(layer_units[i] + 1, layer_units[i+1], n_samples)
        latent['w'+str(i)] = get_output(layer[1])
        given['qv'+str(i)] = layer[0]
        given['pv'+str(i)] = layer[2]
    latent_k, latent_v = map(list, zip(*six.iteritems(latent)))
    latent_v = dict(zip(latent_k, map(lambda x: x[0], latent_v)))

    lower_bound = tf.reduce_mean(advi(model, observed, latent,
                                      given=given, reduction_indices=1))
    grads = optimizer.compute_gradients(-lower_bound)
    infer = optimizer.apply_gradients(grads)

    pwq = []
    for i in range(layer_Num - 1):
        pwq.append(model._multiply(given['pv'+str(i)],
                   latent_v['w'+str(i)], given['qv'+str(i)]))
    rmse, ll = model.evaluate(pwq, observed, std_y_train)

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
                print('>> Test log likelihood = {}'.format(test_ll))
