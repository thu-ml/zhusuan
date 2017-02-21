#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import sys
import os
import time

import tensorflow as tf
from tensorflow.contrib import layers
from six.moves import range
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import zhusuan as zs

import dataset


@zs.reuse('model')
def sbn(observed, n, n_x, n_z, n_particles, is_training):
    with zs.StochasticGraph(observed=observed) as model:
        z_mean = tf.zeros([n_particles, n_z])
        z = zs.Bernoulli('z', z_mean, sample_dim=1, n_samples=n)
        lh_z = layers.fully_connected(z, n_z, activation_fn=None)
        h2 = zs.Bernoulli('h2', lh_z)
        lh_h = layers.fully_connected(h2, n_z, activation_fn=None)
        h1 = zs.Bernoulli('h1', lh_h)
        lx_h = layers.fully_connected(h1, n_x, activation_fn=None)
        x = zs.Bernoulli('x', lx_h)
    return model


def q_net(x, n_z, n_particles, is_training):
    with zs.StochasticGraph() as variational:
        lh_x = layers.fully_connected(x, n_z, activation_fn=None)
        h1 = zs.Bernoulli('h1', lh_x, sample_dim=0, n_samples=n_particles)
        lh_h = layers.fully_connected(h1, n_z, activation_fn=None)
        h2 = zs.Bernoulli('h2', lh_h)
        lz_h = layers.fully_connected(h2, n_z, activation_fn=None)
        z = zs.Bernoulli('z', lz_h)

    return variational

if __name__ == "__main__":
    tf.set_random_seed(1237)

    # Load MNIST
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'data', 'mnist.pkl.gz')
    x_train, t_train, x_valid, t_valid, x_test, t_test = \
        dataset.load_mnist_realval(data_path)
    x_train = np.vstack([x_train, x_valid]).astype('float32')
    np.random.seed(1234)
    x_test = np.random.binomial(1, x_test, size=x_test.shape).astype('float32')
    n_x = x_train.shape[1]

    # Define model parameters
    n_z = 200

    # Define training/evaluation parameters
    lb_samples = 2
    ll_samples = 1000
    epoches = 3000
    batch_size = 24
    test_batch_size = 100
    iters = x_train.shape[0] // batch_size
    test_iters = x_test.shape[0] // test_batch_size
    test_freq = 10
    learning_rate = 0.001
    anneal_lr_freq = 200
    anneal_lr_rate = 0.75

    # Build the computation graph
    is_training = tf.placeholder(tf.bool, shape=[], name='is_training')
    n_particles = tf.placeholder(tf.int32, shape=[], name='n_particles')

    x_orig = tf.placeholder(tf.float32, shape=[None, n_x], name='x')
    x_bin = tf.cast(tf.less(tf.random_uniform(tf.shape(x_orig), 0, 1), x_orig),
                    tf.float32)
    x = tf.placeholder(tf.float32, shape=[None, n_x], name='x')
    x_obs = tf.tile(tf.expand_dims(x, 0), [n_particles, 1, 1])
    n = tf.shape(x)[0]

    def log_joint(observed):
        model = sbn(observed, n, n_x, n_z, n_particles, is_training)
        log_pz, log_ph2_z, log_ph1_h2, log_px_h1 = model.local_log_prob(['z', 'h2', 'h1', 'x'])
        return tf.reduce_sum(log_pz, -1) + tf.reduce_sum(log_ph2_z, -1) + tf.reduce_sum(log_ph1_h2, -1) \
                + tf.reduce_sum(log_px_h1, -1)

    variational = q_net(x, n_z, n_particles, is_training)
    qz_samples, log_qz = variational.query('z', outputs=True,
                                           local_log_prob=True)
    qh1_samples, log_qh1 = variational.query('h1', outputs=True,
                                           local_log_prob=True)
    qh2_samples, log_qh2 = variational.query('h2', outputs=True,
                                           local_log_prob=True)
    log_qz = tf.reduce_sum(log_qz, -1)
    log_qh1 = tf.reduce_sum(log_qh1, -1)
    log_qh2 = tf.reduce_sum(log_qh2, -1)

    object_function, lower_bound = zs.vimco(
        log_joint, {'x': x_obs}, {'z': [qz_samples, log_qz], 'h1': [qh1_samples, log_qh1],
                          'h2': [qh2_samples, log_qh2]}, axis=0, is_particle_larger_one=True)
    lower_bound = tf.reduce_mean(lower_bound)
    object_function = tf.reduce_mean(object_function)
    log_likelihood = tf.reduce_mean(zs.is_loglikelihood(
        log_joint, {'x': x_obs}, {'z': [qz_samples, log_qz], 'h1': [qh1_samples, log_qh1],
                          'h2': [qh2_samples, log_qh2]}, axis=0))

    learning_rate_ph = tf.placeholder(tf.float32, shape=[], name='lr')
    optimizer = tf.train.AdamOptimizer(learning_rate_ph, epsilon=1e-4)
    grads = optimizer.compute_gradients(-object_function)
    infer = optimizer.apply_gradients(grads)

    # train_writer = tf.train.SummaryWriter('/tmp/zhusuan',
    #                                       tf.get_default_graph())
    # train_writer.close()

    params = tf.trainable_variables()
    for i in params:
        print(i.name, i.get_shape())

    # logTrainFile = open('vimco_log_K_2_train.csv', 'w')
    # logTestFile = open('vimco_log_K_2_test.csv', 'w')
    # Run the inference
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(1, epoches + 1):
            time_epoch = -time.time()
            if epoch % anneal_lr_freq == 0:
                learning_rate *= anneal_lr_rate
            np.random.shuffle(x_train)
            lbs = []
            for t in range(iters):
                x_batch = x_train[t * batch_size:(t + 1) * batch_size]
                x_batch_bin = sess.run(x_bin, feed_dict={x_orig: x_batch})
                _, lb = sess.run([infer, lower_bound],
                                 feed_dict={x: x_batch_bin,
                                            learning_rate_ph: learning_rate,
                                            n_particles: lb_samples,
                                            is_training: True})
                lbs.append(lb)
            time_epoch += time.time()
            # logTrainFile.write('%d,%lf\n' % (epoch, np.mean(lbs)))
            print('Epoch {} ({:.1f}s): Lower bound = {}'.format(
                epoch, time_epoch, np.mean(lbs)))
            if epoch % test_freq == 0:
                time_test = -time.time()
                test_lbs = []
                test_lls = []
                for t in range(test_iters):
                    test_x_batch = x_test[
                        t * test_batch_size: (t + 1) * test_batch_size]
                    test_lb = sess.run(lower_bound,
                                       feed_dict={x: test_x_batch,
                                                  n_particles: lb_samples,
                                                  is_training: False})
                    test_ll = sess.run(log_likelihood,
                                       feed_dict={x: test_x_batch,
                                                  n_particles: ll_samples,
                                                  is_training: False})
                    test_lbs.append(test_lb)
                    test_lls.append(test_ll)
                time_test += time.time()
                # logTestFile.write('%lf,%lf\n' % (np.mean(test_lb), np.mean(test_ll)))
                print('>>> TEST ({:.1f}s)'.format(time_test))
                print('>> Test lower bound = {}'.format(np.mean(test_lbs)))
                print('>> Test log likelihood = {}'.format(np.mean(test_lls)))

    # logTestFile.close()
    # logTrainFile.close()