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


@zs.meta_bayesian_net(scope="gen", reuse_variables=True)
def build_gen(n, x_dim, z_dim, n_particles):
    bn = zs.BayesianNet()
    z_mean = tf.zeros([n, z_dim])
    z = bn.normal("z", z_mean, std=1., group_ndims=1, n_samples=n_particles)
    h = tf.layers.dense(z, 500, activation=tf.nn.relu)
    h = tf.layers.dense(h, 500, activation=tf.nn.relu)
    x_logits = tf.layers.dense(h, x_dim)
    bn.bernoulli("x", x_logits, group_ndims=1)
    return bn


@zs.reuse_variables(scope="q_net")
def build_q_net(x, z_dim, n_particles):
    bn = zs.BayesianNet()
    h = tf.layers.dense(tf.cast(x, tf.float32), 500, activation=tf.nn.relu)
    h = tf.layers.dense(h, 500, activation=tf.nn.relu)
    z_mean = tf.layers.dense(h, z_dim)
    z_logstd = tf.layers.dense(h, z_dim)
    bn.normal("z", z_mean, logstd=z_logstd, group_ndims=1,
              n_samples=n_particles)
    return bn


def main():
    tf.set_random_seed(1234)
    np.random.seed(1234)

    # Load MNIST
    data_path = os.path.join(conf.data_dir, "mnist.pkl.gz")
    x_train, t_train, x_valid, t_valid, x_test, t_test = \
        dataset.load_mnist_realval(data_path)
    x_train = np.vstack([x_train, x_valid])
    x_test = np.random.binomial(1, x_test, size=x_test.shape)
    x_dim = x_train.shape[1]

    # Define model parameters
    z_dim = 40

    # Build the computation graph
    n_particles = tf.placeholder(tf.int32, shape=[], name="n_particles")
    x_input = tf.placeholder(tf.float32, shape=[None, x_dim], name="x")
    x = tf.cast(tf.less(tf.random_uniform(tf.shape(x_input)), x_input),
                tf.int32)
    n = tf.placeholder(tf.int32, shape=[], name="n")

    model = build_gen(n, x_dim, z_dim, n_particles)
    variational = build_q_net(x, z_dim, n_particles)

    lower_bound = zs.variational.importance_weighted_objective(
        model, {'x': x}, variational=variational, axis=0)
    cost = tf.reduce_mean(lower_bound.sgvb())
    lower_bound = tf.reduce_mean(lower_bound)

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    infer_op = optimizer.minimize(cost)

    # Define training/evaluation parameters
    lb_samples = 50
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
                                            n_particles: lb_samples,
                                            n: batch_size})
                lbs.append(lb)
            time_epoch += time.time()
            print("Epoch {} ({:.1f}s): IWAE bound = {}".format(
                epoch, time_epoch, np.mean(lbs)))

            if epoch % test_freq == 0:
                time_test = -time.time()
                test_lbs = []
                for t in range(test_iters):
                    test_x_batch = x_test[t * test_batch_size:
                                          (t + 1) * test_batch_size]
                    test_lb = sess.run(lower_bound,
                                       feed_dict={x: test_x_batch,
                                                  n_particles: lb_samples,
                                                  n: test_batch_size})
                    test_lbs.append(test_lb)
                time_test += time.time()
                print(">>> TEST ({:.1f}s)".format(time_test))
                print(">> Test IWAE bound = {}".format(np.mean(test_lbs)))


if __name__ == "__main__":
    main()
