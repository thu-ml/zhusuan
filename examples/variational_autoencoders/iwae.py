#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Importance Weighted Autoencoders, (Burda, 2015)
"""

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
def vae(observed, n, n_x, n_z, n_particles):
    with zs.BayesianNet(observed=observed) as model:
        z_mean = tf.zeros([n, n_z])
        z = zs.Normal('z', z_mean, std=1., n_samples=n_particles,
                      group_ndims=1)
        lx_z = tf.layers.dense(z, 500, activation=tf.nn.relu)
        lx_z = tf.layers.dense(lx_z, 500, activation=tf.nn.relu)
        x_logits = tf.layers.dense(lx_z, n_x)
        x = zs.Bernoulli('x', x_logits, group_ndims=1)
    return model


@zs.reuse('variational')
def q_net(observed, x, n_z, n_particles):
    with zs.BayesianNet(observed=observed) as variational:
        lz_x = tf.layers.dense(tf.to_float(x), 500, activation=tf.nn.relu)
        lz_x = tf.layers.dense(lz_x, 500, activation=tf.nn.relu)
        lz_mean = tf.layers.dense(lz_x, n_z)
        lz_logstd = tf.layers.dense(lz_x, n_z)
        z = zs.Normal('z', lz_mean, logstd=lz_logstd, n_samples=n_particles,
                      group_ndims=1)
    return variational


def main():
    tf.set_random_seed(1237)

    # Load MNIST
    data_path = os.path.join(conf.data_dir, 'mnist.pkl.gz')
    x_train, t_train, x_valid, t_valid, x_test, t_test = \
        dataset.load_mnist_realval(data_path)
    x_train = np.vstack([x_train, x_valid]).astype('float32')
    np.random.seed(1234)
    x_test = np.random.binomial(1, x_test, size=x_test.shape).astype('float32')
    n_x = x_train.shape[1]

    # Define model parameters
    n_z = 40

    # Define training/evaluation parameters
    lb_samples = 50
    epochs = 3000
    batch_size = 1000
    iters = x_train.shape[0] // batch_size
    learning_rate = 0.001
    anneal_lr_freq = 200
    anneal_lr_rate = 0.75
    test_freq = 10
    test_batch_size = 400
    test_iters = x_test.shape[0] // test_batch_size

    # Build the computation graph
    n_particles = tf.placeholder(tf.int32, shape=[], name='n_particles')
    x_orig = tf.placeholder(tf.float32, shape=[None, n_x], name='x')
    x_bin = tf.cast(tf.less(tf.random_uniform(tf.shape(x_orig), 0, 1), x_orig),
                    tf.int32)
    x = tf.placeholder(tf.int32, shape=[None, n_x], name='x')
    x_obs = tf.tile(tf.expand_dims(x, 0), [n_particles, 1, 1])
    n = tf.shape(x)[0]

    def log_joint(observed):
        model = vae(observed, n, n_x, n_z, n_particles)
        log_pz, log_px_z = model.local_log_prob(['z', 'x'])
        return log_pz + log_px_z

    variational = q_net({}, x, n_z, n_particles)
    qz_samples, log_qz = variational.query('z', outputs=True,
                                           local_log_prob=True)
    lower_bound = zs.variational.importance_weighted_objective(
        log_joint, {'x': x_obs}, {'z': [qz_samples, log_qz]}, axis=0)
    cost = tf.reduce_mean(lower_bound.sgvb())
    lower_bound = tf.reduce_mean(lower_bound)

    learning_rate_ph = tf.placeholder(tf.float32, shape=[], name='lr')
    optimizer = tf.train.AdamOptimizer(learning_rate_ph, epsilon=1e-4)
    infer_op = optimizer.minimize(cost)

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
                _, lb = sess.run([infer_op, lower_bound],
                                 feed_dict={x: x_batch_bin,
                                            learning_rate_ph: learning_rate,
                                            n_particles: lb_samples})
                lbs.append(lb)
            time_epoch += time.time()
            print('Epoch {} ({:.1f}s): Lower bound = {}'.format(
                epoch, time_epoch, np.mean(lbs)))

            if epoch % test_freq == 0:
                time_test = -time.time()
                test_lbs = []
                for t in range(test_iters):
                    test_x_batch = x_test[t * test_batch_size:
                                          (t + 1) * test_batch_size]
                    test_lb = sess.run(lower_bound,
                                       feed_dict={x: test_x_batch,
                                                  n_particles: lb_samples})
                    test_lbs.append(test_lb)
                time_test += time.time()
                print('>>> TEST ({:.1f}s)'.format(time_test))
                print('>> Test IWAE bound = {}'.format(np.mean(test_lbs)))


if __name__ == "__main__":
    main()
