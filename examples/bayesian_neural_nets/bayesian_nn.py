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

from examples import conf
from examples.utils import dataset


@zs.meta_bayesian_net(scope="bnn", reuse_variables=True)
def build_bnn(layer_sizes, n_particles):
    bn = zs.BayesianNet()
    x = bn.input("x")
    h = tf.tile(x[None, ...], [n_particles, 1, 1])
    for i, n_out in enumerate(layer_sizes[1:]):
        activation = tf.nn.relu if i < len(layer_sizes) - 2 else None
        h = zs.nn.dense(bn, h, n_out, name=str(i), activation=activation)

    y_mean = bn.output("y_mean", tf.squeeze(h, 2))
    y_logstd = tf.get_variable("y_logstd", shape=[],
                               initializer=tf.constant_initializer(0.))
    bn.normal("y", y_mean, logstd=y_logstd)
    return bn


def main():
    tf.set_random_seed(1234)
    np.random.seed(1234)

    # Load UCI Boston housing data
    data_path = os.path.join(conf.data_dir, "housing.data")
    x_train, y_train, x_valid, y_valid, x_test, y_test = \
        dataset.load_uci_boston_housing(data_path)
    x_train = np.vstack([x_train, x_valid])
    y_train = np.hstack([y_train, y_valid])
    n_train, x_dim = x_train.shape

    # Standardize data
    x_train, x_test, _, _ = dataset.standardize(x_train, x_test)
    y_train, y_test, mean_y_train, std_y_train = dataset.standardize(
        y_train, y_test)

    # Define model parameters
    n_hiddens = [50]

    # Build the computation graph
    n_particles = tf.placeholder(tf.int32, shape=[], name="n_particles")
    x = tf.placeholder(tf.float32, shape=[None, x_dim])
    y = tf.placeholder(tf.float32, shape=[None])
    layer_sizes = [x_dim] + n_hiddens + [1]
    w_names = ["w" + str(i) for i in range(len(layer_sizes) - 1)]

    meta_model = build_bnn(layer_sizes, n_particles)
    variational = zs.nn.mean_field_for_dense_weights()

    def log_joint(bn):
        log_pws = bn.cond_log_prob(w_names)
        log_py_xw = bn.cond_log_prob('y')
        return tf.add_n(log_pws) + tf.reduce_mean(log_py_xw, 1) * n_train

    meta_model.log_joint = log_joint

    lower_bound = zs.variational.elbo(
        meta_model, {'x': x, 'y': y}, variational=variational, axis=0)
    cost = lower_bound.sgvb()

    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    infer_op = optimizer.minimize(cost)

    # prediction: rmse & log likelihood
    y_mean = lower_bound.bn["y_mean"]
    y_pred = tf.reduce_mean(y_mean, 0)
    rmse = tf.sqrt(tf.reduce_mean((y_pred - y) ** 2)) * std_y_train
    log_py_xw = lower_bound.bn.cond_log_prob("y")
    log_likelihood = tf.reduce_mean(zs.log_mean_exp(log_py_xw, 0)) - tf.log(
        std_y_train)

    # Define training/evaluation parameters
    lb_samples = 10
    ll_samples = 5000
    epochs = 500
    batch_size = 10
    iters = (n_train-1) // batch_size + 1
    test_freq = 10

    # Run the inference
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(1, epochs + 1):
            perm = np.random.permutation(x_train.shape[0])
            x_train = x_train[perm, :]
            y_train = y_train[perm]
            lbs = []
            for t in range(iters):
                x_batch = x_train[t * batch_size:(t + 1) * batch_size]
                y_batch = y_train[t * batch_size:(t + 1) * batch_size]
                _, lb = sess.run(
                    [infer_op, lower_bound],
                    feed_dict={n_particles: lb_samples,
                               x: x_batch, y: y_batch})
                lbs.append(lb)
            print('Epoch {}: Lower bound = {}'.format(epoch, np.mean(lbs)))

            if epoch % test_freq == 0:
                test_rmse, test_ll = sess.run(
                    [rmse, log_likelihood],
                    feed_dict={n_particles: ll_samples,
                               x: x_test, y: y_test})
                print('>> TEST')
                print('>> Test rmse = {}, log_likelihood = {}'
                      .format(test_rmse, test_ll))


if __name__ == "__main__":
    main()
