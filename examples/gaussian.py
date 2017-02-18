#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import sys
import os

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import tensorflow as tf
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import zhusuan as zs


def gaussian(observed, n_x, log_stdev, n_particles):
    with zs.StochasticGraph(observed=observed) as model:
        x_mean = tf.zeros([n_particles, n_x])
        x_logstd = log_stdev * tf.ones([n_particles, n_x])
        lx = zs.Normal('x', x_mean, x_logstd)
    return model


if __name__ == "__main__":
    tf.set_random_seed(1)

    # Define model parameters
    n_x = 1
    # n_x = 10
    stdev = 1 / (np.arange(n_x) + 1)
    log_stdev = np.log(stdev)

    # Define HMC parameters
    kernel_width = 0.1
    n_chains = 1000
    n_samples = 200
    burnin = n_samples // 2
    n_leapfrogs = 5

    # Build the computation graph
    def log_joint(observed):
        model = gaussian(observed, n_x, log_stdev, n_chains)
        log_p = model.local_log_prob('x')
        return tf.reduce_sum(log_p, -1)

    adapt_step_size = tf.placeholder(tf.bool, shape=[], name="adapt_step_size")
    adapt_mass = tf.placeholder(tf.bool, shape=[], name="adapt_mass")
    hmc = zs.HMC(step_size=1e-3, n_leapfrogs=n_leapfrogs,
                 adapt_step_size=adapt_step_size, adapt_mass=adapt_mass)
    x = tf.Variable(tf.zeros([n_chains, n_x]), name='x')
    sampler = hmc.sample(log_joint, {}, {'x': x}, chain_axis=0)

    train_writer = tf.summary.FileWriter('/tmp/gaussian',
                                         tf.get_default_graph())
    train_writer.close()

    # Run the inference
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        samples = []
        print('Sampling...')
        for i in range(n_samples):
            q, p, oh, nh, ol, nl, ar, ss = sess.run(
                sampler, feed_dict={adapt_step_size: i < burnin,
                                    adapt_mass: i < burnin})
            print('Sample {}: Acceptance rate = {}, step size = {}'.format(
                i, np.mean(ar), ss))
            if i >= burnin:
                samples.append(q[0])
        print('Finished.')
        samples = np.vstack(samples)

    # Check & plot the results
    def kde(xs, mu, batch_size):
        mu_n = len(mu)
        assert mu_n % batch_size == 0
        xs_row = np.expand_dims(xs, 1)
        ys = np.zeros(xs.shape)
        for b in range(mu_n // batch_size):
            mu_col = np.expand_dims(mu[b * batch_size:(b + 1) * batch_size], 0)
            ys += (1 / np.sqrt(2 * np.pi) / kernel_width) * \
                np.mean(np.exp((-0.5 / kernel_width ** 2) *
                        np.square(xs_row - mu_col)), 1)
        ys /= (mu_n / batch_size)
        return ys

    if n_x == 1:
       xs = np.linspace(-5, 5, 1000)
       ys = kde(xs, np.squeeze(samples), n_chains)
       f, ax = plt.subplots()
       ax.plot(xs, ys)
       ax.plot(xs, stats.norm.pdf(xs, scale=stdev[0]))

    for i in range(n_x):
        print(stats.normaltest(samples[:, i]))

    print('Expected mean = {}'.format(np.zeros(n_x)))
    print('Sample mean = {}'.format(np.mean(samples, 0)))
    print('Expected stdev = {}'.format(stdev))
    print('Sample stdev = {}'.format(np.std(samples, 0)))
    print('Relative error of stdev = {}'.format(
        (np.std(samples, 0) - stdev) / stdev))

    if n_x == 1:
        plt.show()
