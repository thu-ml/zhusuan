#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import time

import tensorflow as tf
from six.moves import range
import numpy as np
import zhusuan as zs

from examples import conf
from examples.utils import dataset


@zs.reuse('model')
def vae(observed, n, x_dim, z_dim, n_particles, is_training):
    with zs.BayesianNet(observed=observed) as model:
        z_logits = tf.zeros([n, z_dim])
        z = zs.Bernoulli('z', z_logits, group_ndims=1, n_samples=n_particles,
                         dtype=tf.float32)
        lx_z = tf.layers.dense(z, 500, use_bias=False)
        lx_z = tf.layers.batch_normalization(lx_z, training=is_training)
        lx_z = tf.nn.relu(lx_z)
        lx_z = tf.layers.dense(lx_z, 500, use_bias=False)
        lx_z = tf.layers.batch_normalization(lx_z, training=is_training)
        lx_z = tf.nn.relu(lx_z)
        x_logits = tf.layers.dense(lx_z, x_dim)
        x = zs.Bernoulli('x', x_logits, group_ndims=1)
    return model


def q_net(x, z_dim, n_particles, is_training):
    with zs.BayesianNet() as variational:
        lz_x = tf.layers.dense(tf.to_float(x), 500, use_bias=False)
        lz_x = tf.layers.batch_normalization(lz_x, training=is_training)
        lz_x = tf.nn.relu(lz_x)
        lz_x = tf.layers.dense(lz_x, 500, use_bias=False)
        lz_x = tf.layers.batch_normalization(lz_x, training=is_training)
        lz_x = tf.nn.relu(lz_x)
        z_logits = tf.layers.dense(lz_x, z_dim)
        z = zs.Bernoulli('z', z_logits, group_ndims=1, n_samples=n_particles,
                         dtype=tf.float32)
    return variational


def baseline_net(x):
    lc_x = tf.layers.dense(tf.to_float(x), 100, activation=tf.nn.relu)
    lc_x = tf.layers.dense(lc_x, 1)
    lc_x = tf.squeeze(lc_x, -1)
    return lc_x


def main():
    # Load MNIST
    data_path = os.path.join(conf.data_dir, 'mnist.pkl.gz')
    x_train, t_train, x_valid, t_valid, x_test, t_test = \
        dataset.load_mnist_realval(data_path)
    x_train = np.vstack([x_train, x_valid])
    x_test = np.random.binomial(1, x_test, size=x_test.shape)
    x_dim = x_train.shape[1]

    # Define model parameters
    z_dim = 40

    # Build the computation graph
    is_training = tf.placeholder(tf.bool, shape=[], name='is_training')
    n_particles = tf.placeholder(tf.int32, shape=[], name='n_particles')
    x_input = tf.placeholder(tf.float32, shape=[None, x_dim], name='x')
    x = tf.to_int32(tf.less(tf.random_uniform(tf.shape(x_input)), x_input))
    n = tf.shape(x)[0]

    def log_joint(observed):
        model = vae(observed, n, x_dim, z_dim, n_particles, is_training)
        log_pz, log_px_z = model.local_log_prob(['z', 'x'])
        return log_pz + log_px_z

    variational = q_net(x, z_dim, n_particles, is_training)
    qz_samples, log_qz = variational.query('z', outputs=True,
                                           local_log_prob=True)
    cx = tf.expand_dims(baseline_net(x), 0)
    lower_bound = zs.variational.elbo(log_joint,
                                      observed={'x': x},
                                      latent={'z': [qz_samples, log_qz]},
                                      axis=0)
    cost, baseline_cost = lower_bound.reinforce(baseline=cx)
    cost = tf.reduce_mean(cost + baseline_cost)
    lower_bound = tf.reduce_mean(lower_bound)

    # Importance sampling estimates of marginal log likelihood
    is_log_likelihood = tf.reduce_mean(
        zs.is_loglikelihood(log_joint, {'x': x},
                            {'z': [qz_samples, log_qz]}, axis=0))

    optimizer = tf.train.AdamOptimizer(0.001)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        infer_op = optimizer.minimize(cost)

    # Define training/evaluation parameters
    epochs = 3000
    batch_size = 128
    iters = x_train.shape[0] // batch_size
    test_freq = 10
    test_batch_size = 400
    test_iters = x_test.shape[0] // test_batch_size

    # Run the inference
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(1, epochs + 1):
            time_epoch = -time.time()
            np.random.shuffle(x_train)
            lbs = []
            for t in range(iters):
                x_batch = x_train[t * batch_size:(t + 1) * batch_size]
                _, lb = sess.run([infer_op, lower_bound],
                                 feed_dict={x_input: x_batch,
                                            is_training: True,
                                            n_particles: 1})
                lbs.append(lb)
            time_epoch += time.time()
            print('Epoch {} ({:.1f}s): Lower bound = {}'.format(
                epoch, time_epoch, np.mean(lbs)))

            if epoch % test_freq == 0:
                time_test = -time.time()
                test_lbs = []
                test_lls = []
                for t in range(test_iters):
                    test_x_batch = x_test[t * test_batch_size:
                                          (t + 1) * test_batch_size]
                    test_lb = sess.run(lower_bound,
                                       feed_dict={x: test_x_batch,
                                                  is_training: False,
                                                  n_particles: 1})
                    test_ll = sess.run(is_log_likelihood,
                                       feed_dict={x: test_x_batch,
                                                  is_training: False,
                                                  n_particles: 1000})
                    test_lbs.append(test_lb)
                    test_lls.append(test_ll)
                time_test += time.time()
                print('>>> TEST ({:.1f}s)'.format(time_test))
                print('>> Test lower bound = {}'.format(np.mean(test_lbs)))
                print('>> Test log likelihood (IS) = {}'.format(
                    np.mean(test_lls)))


if __name__ == "__main__":
    main()
