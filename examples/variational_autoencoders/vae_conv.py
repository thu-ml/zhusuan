#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import time

import tensorflow as tf
from tensorflow.contrib import layers
from six.moves import range
import numpy as np
import zhusuan as zs

from examples import conf
from examples.utils import dataset


@zs.reuse('model')
def vae_conv(observed, n, n_x, n_z, n_particles, is_training):
    with zs.BayesianNet(observed=observed) as model:
        normalizer_params = {'is_training': is_training,
                             'updates_collections': None}
        z_mean = tf.zeros([n, n_z])
        z = zs.Normal('z', z_mean, std=1., n_samples=n_particles,
                      group_event_ndims=1)
        lx_z = tf.reshape(z, [-1, 1, 1, n_z])
        lx_z = layers.conv2d_transpose(
            lx_z, 128, kernel_size=3, padding='VALID',
            normalizer_fn=layers.batch_norm,
            normalizer_params=normalizer_params)
        lx_z = layers.conv2d_transpose(
            lx_z, 64, kernel_size=5, padding='VALID',
            normalizer_fn=layers.batch_norm,
            normalizer_params=normalizer_params)
        lx_z = layers.conv2d_transpose(
            lx_z, 32, kernel_size=5, stride=2,
            normalizer_fn=layers.batch_norm,
            normalizer_params=normalizer_params)
        lx_z = layers.conv2d_transpose(
            lx_z, 1, kernel_size=5, stride=2,
            activation_fn=None)
        x_logits = tf.reshape(lx_z, [n_particles, n, -1])
        x = zs.Bernoulli('x', x_logits, group_event_ndims=1)
    return model


def q_net(x, n_xl, n_z, n_particles, is_training):
    with zs.BayesianNet() as variational:
        normalizer_params = {'is_training': is_training,
                             'updates_collections': None}
        lz_x = tf.reshape(tf.to_float(x), [-1, n_xl, n_xl, 1])
        lz_x = layers.conv2d(
            lz_x, 32, kernel_size=5, stride=2,
            normalizer_fn=layers.batch_norm,
            normalizer_params=normalizer_params)
        lz_x = layers.conv2d(
            lz_x, 64, kernel_size=5, stride=2,
            normalizer_fn=layers.batch_norm,
            normalizer_params=normalizer_params)
        lz_x = layers.conv2d(
            lz_x, 128, kernel_size=5, padding='VALID',
            normalizer_fn=layers.batch_norm,
            normalizer_params=normalizer_params)
        lz_x = layers.dropout(lz_x, keep_prob=0.9, is_training=is_training)
        lz_x = tf.reshape(lz_x, [-1, 128 * 3 * 3])
        lz_mean = layers.fully_connected(lz_x, n_z, activation_fn=None)
        lz_logstd = layers.fully_connected(lz_x, n_z, activation_fn=None)
        z = zs.Normal('z', lz_mean, logstd=lz_logstd, n_samples=n_particles,
                      group_event_ndims=1)
    return variational


if __name__ == "__main__":
    tf.set_random_seed(1237)

    # Load MNIST
    data_path = os.path.join(conf.data_dir, 'mnist.pkl.gz')
    x_train, t_train, x_valid, t_valid, x_test, t_test = \
        dataset.load_mnist_realval(data_path)
    x_train = np.vstack([x_train, x_valid]).astype('float32')
    np.random.seed(1234)
    x_test = np.random.binomial(1, x_test, size=x_test.shape).astype('float32')
    n_x = x_train.shape[1]
    n_xl = int(np.sqrt(n_x))

    # Define model parameters
    n_z = 40

    # Define training/evaluation parameters
    lb_samples = 1
    ll_samples = 100
    epochs = 3000
    batch_size = 100
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
                    tf.int32)
    x = tf.placeholder(tf.int32, shape=[None, n_x], name='x')
    x_obs = tf.tile(tf.expand_dims(x, 0), [n_particles, 1, 1])
    n = tf.shape(x)[0]

    def log_joint(observed):
        model = vae_conv(observed, n, n_x, n_z, n_particles, is_training)
        log_pz, log_px_z = model.local_log_prob(['z', 'x'])
        return log_pz + log_px_z

    variational = q_net(x, n_xl, n_z, n_particles, is_training)
    qz_samples, log_qz = variational.query('z', outputs=True,
                                           local_log_prob=True)
    lower_bound = tf.reduce_mean(
        zs.sgvb(log_joint, {'x': x_obs}, {'z': [qz_samples, log_qz]}, axis=0))
    log_likelihood = tf.reduce_mean(
        zs.is_loglikelihood(log_joint, {'x': x_obs},
                            {'z': [qz_samples, log_qz]}, axis=0))

    learning_rate_ph = tf.placeholder(tf.float32, shape=[], name='lr')
    optimizer = tf.train.AdamOptimizer(learning_rate_ph, epsilon=1e-4)
    grads = optimizer.compute_gradients(-lower_bound)
    infer = optimizer.apply_gradients(grads)

    params = tf.trainable_variables()
    for i in params:
        print(i.name, i.get_shape())

    # Run the inference
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(1, epochs + 1):
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
                print('>>> TEST ({:.1f}s)'.format(time_test))
                print('>> Test lower bound = {}'.format(np.mean(test_lbs)))
                print('>> Test log likelihood = {}'.format(np.mean(test_lls)))
