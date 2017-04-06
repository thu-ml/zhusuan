#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import sys
import os
import time

import tensorflow as tf
import tensorflow.contrib.layers as layers
from six.moves import range
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import zhusuan as zs

import dataset


@zs.reuse('model')
def var_dropout(observed, x, n, net_size, n_particles, is_training):
    with zs.BayesianNet(observed=observed) as model:
        h = x
        normalizer_params = {'is_training': is_training,
                             'updates_collections': None}
        for i, [n_in, n_out] in enumerate(zip(net_size[:-1], net_size[1:])):
            eps_mean = tf.ones([n, n_in])
            eps_logstd = tf.zeros([n, n_in])
            eps = zs.Normal('layer' + str(i) + '/eps', eps_mean, eps_logstd,
                            n_samples=n_particles, group_event_ndims=1)
            # TODO: there seems to be a bug because h are using relu already
            h = layers.fully_connected(
                h * eps, n_out, normalizer_fn=layers.batch_norm,
                normalizer_params=normalizer_params)
            if i < len(net_size) - 2:
                h = tf.nn.relu(h)
        # TODO: why sampling here?
        y = zs.OnehotCategorical('y', h)
    return model, y


@zs.reuse('variational')
def q(observed, n, net_size, n_particles):
    with zs.BayesianNet(observed=observed) as variational:
        for i, [n_in, n_out] in enumerate(zip(net_size[:-1], net_size[1:])):
            with tf.variable_scope('layer' + str(i)):
                logit_alpha = tf.get_variable('logit_alpha', [n_in])

            alpha = tf.nn.sigmoid(logit_alpha)
            alpha = tf.tile(tf.expand_dims(alpha, 0), [n, 1])
            eps = zs.Normal('layer' + str(i) + '/eps',
                            1., 0.5 * tf.log(alpha + 1e-10),
                            n_samples=n_particles, group_event_ndims=1)
    return variational


if __name__ == '__main__':
    tf.set_random_seed(1234)
    np.random.seed(1234)

    # Load MNIST
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'data', 'mnist.pkl.gz')
    x_train, y_train, x_valid, y_valid, x_test, y_test = \
        dataset.load_mnist_realval(data_path)
    x_train = np.vstack([x_train, x_valid]).astype('float32')
    y_train = np.vstack([y_train, y_valid])
    x_train, x_test, _, _ = dataset.standardize(x_train, x_test)
    n_x = x_train.shape[1]
    n_class = 10

    # Define training/evaluation parameters
    epoches = 500
    batch_size = 1000
    lb_samples = 1
    ll_samples = 100
    iters = int(np.floor(x_train.shape[0] / float(batch_size)))
    test_freq = 3
    learning_rate = 0.001
    anneal_lr_freq = 100
    anneal_lr_rate = 0.75

    # placeholders
    n_particles = tf.placeholder(tf.int32, shape=[], name='n_particles')
    is_training = tf.placeholder(tf.bool, shape=[], name='is_training')
    x = tf.placeholder(tf.float32, shape=(None, n_x))
    y = tf.placeholder(tf.int32, shape=(None, n_class))
    n = tf.shape(x)[0]

    net_size = [n_x, 100, 100, 100, n_class]
    e_names = ['layer' + str(i) + '/eps' for i in range(len(net_size) - 1)]

    x_obs = tf.tile(tf.expand_dims(x, 0), [n_particles, 1, 1])
    y_obs = tf.tile(tf.expand_dims(y, 0), [n_particles, 1, 1])

    def log_joint(observed):
        model, _ = var_dropout(observed, x_obs, n, net_size,
                               n_particles, is_training)
        log_pe = model.local_log_prob(e_names)
        log_py_xe = model.local_log_prob('y')
        return tf.add_n(log_pe) / x_train.shape[0] + log_py_xe

    variational = q({}, n, net_size, n_particles)
    qe_queries = variational.query(e_names, outputs=True, local_log_prob=True)
    qe_samples, log_qes = zip(*qe_queries)
    log_qes = [log_qe / x_train.shape[0] for log_qe in log_qes]
    e_dict = dict(zip(e_names, zip(qe_samples, log_qes)))
    lower_bound = tf.reduce_mean(
        zs.sgvb(log_joint, {'y': y_obs}, e_dict, axis=0))

    with tf.name_scope('accuracy'):
        _, y_pred = var_dropout({}, x_obs, n, net_size,
                                n_particles, is_training)
        y_pred = tf.reduce_mean(y_pred, 0)
        y_pred = tf.argmax(y_pred, 1)
        sparse_y = tf.argmax(y, 1)
        acc = tf.reduce_mean(tf.cast(tf.equal(y_pred, sparse_y), tf.float32))

    learning_rate_ph = tf.placeholder(tf.float32, shape=())
    optimizer = tf.train.AdamOptimizer(learning_rate_ph, epsilon=1e-4)
    infer = optimizer.minimize(-lower_bound)

    # Run the inference
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(1, epoches + 1):
            if epoch % anneal_lr_freq == 0:
                learning_rate *= anneal_lr_rate
            time_epoch = -time.time()
            lbs = []
            for t in range(iters):
                x_batch = x_train[t * batch_size:(t + 1) * batch_size]
                y_batch = y_train[t * batch_size:(t + 1) * batch_size]
                _, lb = sess.run(
                    [infer, lower_bound],
                    feed_dict={n_particles: lb_samples,
                               is_training: True,
                               learning_rate_ph: learning_rate,
                               x: x_batch, y: y_batch})
                lbs.append(lb)
            time_epoch += time.time()
            print('Epoch {} ({:.1f}s): Lower bound = {}'.format(
                epoch, time_epoch, np.mean(lbs)))

            if epoch % test_freq == 0:
                time_test = -time.time()
                test_lbs = []
                test_accs = []
                for t in range(10):
                    x_batch = x_test[t * 1000:(t + 1) * 1000]
                    y_batch = y_test[t * 1000:(t + 1) * 1000]
                    lb, acc1 = sess.run(
                        [lower_bound, acc],
                        feed_dict={n_particles: ll_samples,
                                   is_training: False,
                                   x: x_batch, y: y_batch})
                    test_lbs.append(lb)
                    test_accs.append(acc1)
                time_test += time.time()
                print('>>> TEST ({:.1f}s)'.format(time_test))
                print('>> Test lower bound = {}'.format(np.mean(test_lbs)))
                print('>> Test accuaracy = {}'.format(np.mean(test_accs)))
