#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import sys
import os
import time

import tensorflow as tf
from six.moves import range, zip
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import zhusuan as zs
except:
    raise ImportError()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import dataset
except:
    raise ImportError()


def bayesianNN(observed, x, n_x, layer_sizes, n_particles):
    with zs.StochasticGraph(observed=observed) as model:
        ws = []
        for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1],
                                              layer_sizes[1:])):
            w_mu = tf.zeros([n_particles, n_out, n_in + 1])
            w_logstd = tf.zeros([n_particles, n_out, n_in + 1])
            ws.append(zs.Normal('w' + str(i), w_mu, w_logstd))

        # forward
        ly_x = tf.expand_dims(
            tf.tile(tf.expand_dims(x, 0), [n_particles, 1, 1]), 2)
        for i in range(len(ws)):
            w = tf.tile(tf.expand_dims(ws[i], 1), [1, tf.shape(x)[0], 1, 1])
            ly_x = tf.concat_v2(
                [ly_x, tf.ones([n_particles, tf.shape(x)[0], 1, 1])], 2)
            ly_x = tf.matmul(w, ly_x) / \
                tf.sqrt(tf.cast(tf.shape(x)[1], tf.float32))
            if i < len(ws) - 1:
                ly_x = tf.nn.relu(ly_x)

        y_mean = tf.squeeze(ly_x, [2, 3])
        y_logstd = tf.Variable(0.)
        y = zs.Normal('y', y_mean, y_logstd * tf.ones_like(y_mean))

    return model


def mean_field_variational(layer_sizes, n_particles):
    with zs.StochasticGraph() as variational:
        ws = []
        for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1],
                                              layer_sizes[1:])):
            mu = tf.Variable(tf.zeros([n_out, n_in + 1]))
            logstd = tf.Variable(tf.zeros([n_out, n_in + 1]))
            ws.append(
                zs.Normal('w' + str(i), mu, logstd, sample_dim=0,
                          n_samples=n_particles))
    return variational


def evaluate(latent, observed, std_y_train):
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
    y_train = np.hstack([y_train, y_valid]).astype('float32')
    y_train = y_train.reshape((len(y_train), 1))
    x_test = x_test.astype('float32')
    y_test = y_test.reshape((len(y_test), 1))
    N, n_x = x_train.shape
    x_train, x_test, _, _ = dataset.standardize(x_train, x_test)
    y_train, y_test, mean_y_train, std_y_train = dataset.standardize(
        y_train, y_test)
    std_y_train = np.squeeze(std_y_train)

    # Define model parameters
    n_hiddens = [50]

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

    # Build the computation graph
    learning_rate_ph = tf.placeholder(tf.float32, shape=())
    n_particles = tf.placeholder(tf.int32, shape=[], name='n_particles')
    x = tf.placeholder(tf.float32, shape=(None, n_x))
    y = tf.placeholder(tf.float32, shape=(None, 1))
    optimizer = tf.train.AdamOptimizer(learning_rate_ph, epsilon=1e-4)
    layer_sizes = [n_x] + n_hiddens + [1]
    w_names = ['w' + str(i) for i in range(len(layer_sizes) - 1)]

    def log_joint(latent, observed, given):
        y = tf.squeeze(observed['y'], [1])
        y = tf.tile(tf.expand_dims(y, 0), [n_particles, 1])
        observed = dict((w_name, latent[w_name]) for w_name in w_names)
        observed.update({'y': y})
        model = bayesianNN(observed, x, n_x, layer_sizes, n_particles)
        log_pws = model.local_log_prob(w_names)
        log_py_xw = model.local_log_prob(y)
        return sum([tf.reduce_sum(log_pw, [-1, -2]) for log_pw in log_pws]) + \
            tf.reduce_mean(log_py_xw, 1) * N

    variational = mean_field_variational(layer_sizes, n_particles)
    qw_outputs = variational.query(w_names, outputs=True, local_log_prob=True)
    qw_outputs = [[qw_samples, tf.reduce_sum(log_qw, [-1, -2])] for
                  qw_samples, log_qw in qw_outputs]
    latent = list(zip(w_names, qw_outputs))
    lower_bound = tf.reduce_mean(
        zs.advi(log_joint, {'y': y}, latent, reduction_indices=0))
    grads = optimizer.compute_gradients(-lower_bound)
    infer = optimizer.apply_gradients(grads)

    # evaluation
    model = bayesianNN({})

    def evaluate():
        y = tf.squeeze(observed['y'], [1])
        ws = [latent['w' + str(i)] for i in range(len(self.layer_sizes) - 1)]

        inputs = dict(zip(self.ws, ws))
        y_out, _ = self.model.get_output(self.ly_x, inputs=inputs)
        y_pred = tf.reduce_mean(y_out, 0)
        rmse = tf.sqrt(tf.reduce_mean((y_pred - y) ** 2)) * std_y_train

        mean = tf.reduce_mean(y_out, 0, keep_dims=True)
        variance = tf.reduce_mean((y_out - mean) ** 2, 0)
        variance = variance + tf.exp(self.y_logstd) ** 2
        ll = tf.reduce_mean(-0.5 *
                            tf.log(2 * np.pi * variance * std_y_train ** 2) -
                            0.5 * (y - y_pred) ** 2 / variance)

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
