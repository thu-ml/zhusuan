#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import time

import tensorflow as tf
from six.moves import range, zip
import numpy as np
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__)))))
import zhusuan as zs

from examples import conf
from examples.utils import dataset


@zs.reuse('model')
def bayesianNN(observed, x, n_x, layer_sizes):
    with zs.BayesianNet(observed=observed) as model:
        ws = []
        for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1],
                                              layer_sizes[1:])):
            w_mu = tf.zeros([n_out, n_in + 1])
            w_logstd = tf.zeros([n_out, n_in + 1])
            ws.append(zs.Normal('w' + str(i), w_mu, w_logstd,
                                group_event_ndims=2))

        # forward
        ly_x = tf.transpose(x, [1, 0])
        for i, w in enumerate(ws):
            ly_x = tf.concat(
                [ly_x, tf.ones([1, tf.shape(x)[0]])], 0)
            ly_x = tf.matmul(w, ly_x) / tf.sqrt(tf.cast(tf.shape(ly_x)[0],
                                                        tf.float32))
            if i < len(ws) - 1:
                ly_x = tf.nn.relu(ly_x)

        y_mean = tf.squeeze(tf.transpose(ly_x, [1, 0]), [1])
        y_logstd = tf.get_variable('y_logstd', shape=[],
                                   initializer=tf.constant_initializer(0.))
        y = zs.Normal('y', y_mean, y_logstd)

    return model, y_mean


def mean_field_variational(layer_sizes):
    with zs.BayesianNet() as variational:
        ws = []
        for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1],
                                              layer_sizes[1:])):
            w_mean = tf.get_variable(
                'w_mean_' + str(i), shape=[n_out, n_in + 1],
                initializer=tf.constant_initializer(0.))
            w_logstd = tf.get_variable(
                'w_logstd_' + str(i), shape=[n_out, n_in + 1],
                initializer=tf.constant_initializer(0.))
            ws.append(
                zs.Normal('w' + str(i), w_mean, w_logstd,
                          group_event_ndims=2))
    return variational


if __name__ == '__main__':
    tf.set_random_seed(1237)
    np.random.seed(1234)

    # Load UCI Boston housing data
    data_path = os.path.join(conf.data_dir, 'housing.data')
    x_train, y_train, x_valid, y_valid, x_test, y_test = \
        dataset.load_uci_boston_housing(data_path)
    x_train = np.vstack([x_train, x_valid]).astype('float32')
    y_train = np.hstack([y_train, y_valid]).astype('float32')
    x_test = x_test.astype('float32')
    N, n_x = x_train.shape

    # Standardize data
    x_train, x_test, _, _ = dataset.standardize(x_train, x_test)
    y_train, y_test, mean_y_train, std_y_train = dataset.standardize(
        y_train.reshape((-1, 1)), y_test.reshape((-1, 1)))
    y_train, y_test = y_train.squeeze(), y_test.squeeze()
    std_y_train = std_y_train.squeeze()

    # Define model parameters
    n_hiddens = [50]
    layer_sizes = [n_x] + n_hiddens + [1]
    w_names = ['w' + str(i) for i in range(len(layer_sizes) - 1)]

    # Build the computation graph
    x = tf.placeholder(tf.float32, shape=[None, n_x])
    y = tf.placeholder(tf.float32, shape=[None])

    def log_joint(observed):
        model, _ = bayesianNN(observed, x, n_x, layer_sizes)
        log_pws = model.local_log_prob(w_names)
        log_py_xw = model.local_log_prob('y')
        return tf.add_n(log_pws) + tf.reduce_mean(log_py_xw) * N

    variational = mean_field_variational(layer_sizes)
    qw_outputs = variational.query(w_names, outputs=True, local_log_prob=True)
    latent = dict(zip(w_names, qw_outputs))
    lower_bound = tf.reduce_mean(
        zs.sgvb(log_joint, {'y': y}, latent))

    optimizer = tf.train.AdamOptimizer(0.001, epsilon=1e-4)
    infer = optimizer.minimize(-lower_bound)

    # prediction: rmse & log likelihood
    observed = dict((w_name, latent[w_name][0]) for w_name in w_names)
    observed.update({'y': y})
    model, y_mean = bayesianNN(observed, x, n_x, layer_sizes)
    rmse = tf.sqrt(tf.reduce_mean((y_mean - y) ** 2)) * std_y_train
    log_py_xw = model.local_log_prob('y')
    log_likelihood = tf.reduce_mean(log_py_xw) - tf.log(std_y_train)

    # Define training/evaluation parameters
    epoches = 500
    batch_size = 10
    iters = int(np.floor(x_train.shape[0] / float(batch_size)))
    test_freq = 10

    # Run the inference
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(1, epoches + 1):
            indices = np.random.permutation(N)
            x_train = x_train[indices, :]
            y_train = y_train[indices]
            lbs = []
            for t in range(iters):
                x_batch = x_train[t * batch_size:(t + 1) * batch_size]
                y_batch = y_train[t * batch_size:(t + 1) * batch_size]
                _, lb = sess.run(
                    [infer, lower_bound],
                    feed_dict={x: x_batch, y: y_batch})
                lbs.append(lb)
            print('Epoch {}: Lower bound = {}'.format(
                epoch, np.mean(lbs)))

            if epoch % test_freq == 0:
                test_lb, test_rmse, test_ll = sess.run(
                    [lower_bound, rmse, log_likelihood],
                    feed_dict={x: x_test, y: y_test})
                print('>>> TEST')
                print('>> Test lower bound = {}'.format(test_lb))
                print('>> Test rmse = {}'.format(test_rmse))
                print('>> Test log_likelihood = {}'.format(test_ll))
