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


@zs.meta_bayesian_net(scope="sbn", reuse_variables=True)
def build_sbn(n, x_dim, h_dim, n_particles):
    bn = zs.BayesianNet()
    h3_logits = tf.zeros([n, h_dim])
    h3 = bn.bernoulli("h3", h3_logits, group_ndims=1, n_samples=n_particles,
                      dtype=tf.float32)
    h2_logits = tf.layers.dense(h3, h_dim)
    h2 = bn.bernoulli("h2", h2_logits, group_ndims=1, dtype=tf.float32)
    h1_logits = tf.layers.dense(h2, h_dim)
    h1 = bn.bernoulli("h1", h1_logits, group_ndims=1, dtype=tf.float32)
    x_logits = tf.layers.dense(h1, x_dim)
    bn.bernoulli("x", x_logits, group_ndims=1)
    return bn


@zs.reuse_variables(scope="proposal")
def build_proposal(x, h_dim, n_particles):
    bn = zs.BayesianNet()
    h1_logits = tf.layers.dense(tf.cast(x, tf.float32), h_dim)
    h1 = bn.bernoulli("h1", h1_logits, group_ndims=1,
                      n_samples=n_particles, dtype=tf.float32)
    h2_logits = tf.layers.dense(h1, h_dim)
    h2 = bn.bernoulli("h2", h2_logits, group_ndims=1, dtype=tf.float32)
    h3_logits = tf.layers.dense(h2, h_dim)
    bn.bernoulli("h3", h3_logits, group_ndims=1, dtype=tf.float32)
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
    h_dim = 200

    # Build the computation graph
    n_particles = tf.placeholder(tf.int32, shape=[], name="n_particles")
    x_input = tf.placeholder(tf.float32, shape=[None, x_dim], name="x")
    x = tf.cast(tf.less(tf.random_uniform(tf.shape(x_input)), x_input),
                tf.int32)
    n = tf.placeholder(tf.int32, shape=[], name="n")

    model = build_sbn(n, x_dim, h_dim, n_particles)
    proposal = build_proposal(x, h_dim, n_particles)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001, epsilon=1e-4)

    # learning model parameters
    lower_bound = tf.reduce_mean(
        zs.variational.importance_weighted_objective(
            model, observed={"x": x}, variational=proposal, axis=0))
    model_params = tf.trainable_variables(scope="sbn")
    model_grads = optimizer.compute_gradients(-lower_bound, model_params)

    # adapting the proposal
    klpq_obj = zs.variational.klpq(
        model, observed={"x": x}, variational=proposal, axis=0)
    klpq_cost = tf.reduce_mean(klpq_obj.importance())
    proposal_params = tf.trainable_variables(scope="proposal")
    klpq_grads = optimizer.compute_gradients(klpq_cost, proposal_params)

    infer_op = optimizer.apply_gradients(model_grads + klpq_grads)

    # Define training/evaluation parameters
    lb_samples = 10
    ll_samples = 1000
    epochs = 3000
    batch_size = 24
    iters = x_train.shape[0] // batch_size
    test_freq = 10
    test_batch_size = 100
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
            print("Epoch {} ({:.1f}s): Lower bound = {}".format(
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
                                                  n: test_batch_size})
                    test_ll = sess.run(lower_bound,
                                       feed_dict={x: test_x_batch,
                                                  n_particles: ll_samples,
                                                  n: test_batch_size})
                    test_lbs.append(test_lb)
                    test_lls.append(test_ll)
                time_test += time.time()
                print(">>> TEST ({:.1f}s)".format(time_test))
                print(">> Test lower bound = {}".format(np.mean(test_lbs)))
                print(">> Test log likelihood = {}".format(np.mean(test_lls)))


if __name__ == "__main__":
    main()
