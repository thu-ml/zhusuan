#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os

import tensorflow as tf
from six.moves import range, zip
import numpy as np
import zhusuan as zs
from copy import copy

from examples import conf
from examples.utils import dataset


@zs.reuse('model')
def bayesianNN(observed, x, layer_sizes, n_particles):
    with zs.BayesianNet(observed=observed) as model:
        ws = []
        for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1],
                                              layer_sizes[1:])):
            w_mu = tf.zeros([1, n_out, n_in + 1])
            ws.append(
                zs.Normal('w' + str(i), w_mu, std=1.,
                          n_samples=n_particles, group_ndims=2))

        # forward
        ly_x = tf.expand_dims(
            tf.tile(tf.expand_dims(x, 0), [n_particles, 1, 1]), 3)
        for i in range(len(ws)):
            w = tf.tile(ws[i], [1, tf.shape(x)[0], 1, 1])
            ly_x = tf.concat(
                [ly_x, tf.ones([n_particles, tf.shape(x)[0], 1, 1])], 2)
            ly_x = tf.matmul(w, ly_x) / tf.sqrt(tf.to_float(tf.shape(ly_x)[2]))
            if i < len(ws) - 1:
                ly_x = tf.nn.relu(ly_x)

        y_mean = tf.squeeze(ly_x, [2, 3])
        y_logstd = -0.95
        y = zs.Normal('y', y_mean, logstd=y_logstd)

    return model, y_mean


def main():
    tf.set_random_seed(1237)
    np.random.seed(2345)

    # Load UCI Boston housing data
    data_path = os.path.join(conf.data_dir, 'housing.data')
    x_train, y_train, x_valid, y_valid, x_test, y_test = \
        dataset.load_uci_boston_housing(data_path)
    x_train = np.vstack([x_train, x_valid])
    y_train = np.hstack([y_train, y_valid])
    N, n_x = x_train.shape

    # Standardize data
    x_train, x_test, _, _ = dataset.standardize(x_train, x_test)
    y_train, y_test, mean_y_train, std_y_train = dataset.standardize(
        y_train, y_test)

    # Define model parameters
    n_hiddens = [50]

    # Build the computation graph
    n_particles = 50
    x = tf.placeholder(tf.float32, shape=[None, n_x])
    y = tf.placeholder(tf.float32, shape=[None])
    layer_sizes = [n_x] + n_hiddens + [1]
    w_names = ['w' + str(i) for i in range(len(layer_sizes) - 1)]
    wv = []
    for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1],
                                      layer_sizes[1:])):
        wv.append(tf.Variable(tf.random_uniform([n_particles, 1, n_out, n_in + 1])*2-1))

    def log_joint(observed):
        model, _ = bayesianNN(observed, x, layer_sizes, n_particles)
        log_pws = model.local_log_prob(w_names)
        log_py_xw = model.local_log_prob('y')
        return tf.reduce_mean(tf.add_n(log_pws) + log_py_xw * N, -1)

    hmc = zs.HMC(step_size=0.04, n_leapfrogs=10, adapt_step_size=True)
    latent = dict(zip(w_names, wv))
    sample_op, hmc_info = hmc.sample(log_joint, observed={'y': y}, latent=latent)

    # prediction: rmse & log likelihood
    observed = copy(latent)
    observed.update({'y': y})
    model, y_mean = bayesianNN(observed, x, layer_sizes, n_particles)
    y_pred = tf.reduce_mean(y_mean, 0)
    rmse = tf.sqrt(tf.reduce_mean((y_pred - y) ** 2)) * std_y_train
    log_py_xw = model.local_log_prob('y')
    log_likelihood = tf.reduce_mean(zs.log_mean_exp(log_py_xw, 0)) - \
        tf.log(std_y_train)

    # Define training/evaluation parameters
    epochs = 500
    batch_size = 10
    iters = int(np.floor(x_train.shape[0] / float(batch_size)))

    # Run the inference
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(1, epochs + 1):
            _, acc = sess.run([sample_op, hmc_info.acceptance_rate],
                                  feed_dict={x: x_train, y: y_train})

            test_rmse = sess.run(rmse,
                                 feed_dict={x: x_test, y: y_test})
            print('>> Epoch {}, acc = {}, Test = {}'.format(epoch, np.mean(acc), test_rmse))


if __name__ == "__main__":
    main()
