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
import zhusuan as zs

import dataset
import pdb

@zs.reuse('model')
def var_dropout(observed, x, n_x, n_class):
    with zs.StochasticGraph(observed=observed) as model:
        #TODO:add a constant tensor
        with tf.variable_scope('layer1'):
            h1 = independent_noise_layer(x, n_x, 200)
        with tf.variable_scope('layer2'):
            h2 = independent_noise_layer(h1, 200, 200)
        with tf.variable_scope('layer3'):
            h3 = independent_noise_layer(h2, 200, 200)
        with tf.variable_scope('layer4'):
            h4 = independent_noise_layer(h3, 200, n_class, None)
        y = zs.Discrete('y', h4)
    return model, y

def independent_noise_layer(x, n_in, n_out, activation=tf.nn.relu):
    """variational dropout layer with independent noise"""
    w = tf.get_variable('weights', [n_in, n_out])
    unnormalized_alpha = tf.get_variable('unnormalized_alpha', [])
    alpha = 1 - 1. / (1 + tf.exp(unnormalized_alpha))

    epsilon = tf.random_normal(tf.shape(x), 1., alpha**0.5)
    w = tf.tile(tf.expand_dims(w, 0), [tf.shape(x)[0], 1, 1])

    h = tf.matmul(x*epsilon, w)
    return h if activation is None else activation(h)

def correlated_noise_layer(x, n_in, n_out, activation=tf.nn.relu):
    """variational dropout layer with correlated noise"""
    w = tf.get_variable('weights', [n_in, n_out])
    unnormalized_alpha = tf.get_variable('unnormalized_alpha', [])
    alpha = 1 - 1. / (1 + tf.exp(unnormalized_alpha))

    epsilon = tf.random_normal([tf.shape(x)[0], tf.shape(x)[2]], 1., alpha**0.5)

    w = tf.tile(tf.expand_dims(w, 0), [tf.shape(x)[0], 1, 1])
    epsilon = tf.tile(tf.expand_dims(epsilon, 1), [1, tf.shape(w)[1], 1])

    h = tf.matmul(x, w*epsilon)
    return h if activation is None else activation(h)

if __name__ == '__main__':
    tf.set_random_seed(1234)
    np.random.seed(1234)

    # Load MNIST
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'data', 'mnist.pkl.gz')
    x_train, y_train, x_valid, y_valid, x_test, y_test = \
        dataset.load_mnist_realval(data_path)
    x_train = np.vstack([x_train, x_valid]).astype('float32')
    y_train = np.vstack([y_train, y_valid]).astype('float32')
    x_train, x_test, _, _ = dataset.standardize(x_train, x_test)
    n_x = x_train.shape[1]
    n_class = 10

    # Define training/evaluation parameters
    epoches = 500
    batch_size = 1000
    lb_samples = 1
    ll_samples = 20
    iters = int(np.floor(x_train.shape[0] / float(batch_size)))
    test_freq = 3
    learning_rate = 0.001

    # Build trainging model
    n_particles = tf.placeholder(tf.int32, shape=[], name='n_particles')
    x = tf.placeholder(tf.float32, shape=(None, n_x))
    y = tf.placeholder(tf.float32, shape=(None, n_class))
    n = tf.shape(x)[0]

    x_obs = tf.tile(tf.expand_dims(x, 0), [n_particles, 1, 1])
    y_obs = tf.tile(tf.expand_dims(y, 0), [n_particles, 1, 1])

    with tf.name_scope('cross_entropy'):
        model, _  = var_dropout({'y': y_obs}, x_obs, n_x, n_class)
        log_py_x = model.local_log_prob('y')

    with tf.name_scope('KL_divergence'):
        def KL(unnormalized_alpha):
            alpha = 1 - 1. / (1 + tf.exp(unnormalized_alpha))
            c1, c2, c3 = (1.16145124, -1.16145124, 1.16145124)
            return -tf.reduce_sum(0.5 * tf.log(alpha) \
                + c1*alpha + c2*alpha**2 + c3*alpha**3)
        KL_div = sum([KL(a) for a in tf.trainable_variables() \
                     if a.name.find('alpha')>=0])

    with tf.name_scope('accuracy'):
        _, y_pred = var_dropout({}, x_obs, n_x, n_class)
        y_pred = tf.reduce_mean(y_pred, 0)
        y_pred = tf.argmax(y_pred, 1)
        y0 = tf.argmax(y, 1)
        acc = tf.reduce_mean(tf.cast(tf.equal(y_pred, y0), tf.float32))

    lower_bound = tf.reduce_mean(log_py_x - KL_div)
    learning_rate_ph = tf.placeholder(tf.float32, shape=())
    optimizer = tf.train.AdamOptimizer(learning_rate_ph, epsilon=1e-4)
    infer = optimizer.minimize(-lower_bound)

    params = tf.trainable_variables()
    for i in params:
        print(i.name, i.get_shape())

    # Run the inference
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(1, epoches + 1):
            time_epoch = -time.time()
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
                test_lb, test_acc = sess.run(
                    [lower_bound, acc],
                    feed_dict={n_particles: ll_samples,
                               x: x_test, y: y_test})
                time_test += time.time()
                print('>>> TEST ({:.1f}s)'.format(time_test))
                print('>> Test lower bound = {}'.format(test_lb))
                print('>> Test accuaracy = {}'.format(test_acc))
