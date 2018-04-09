#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("tkAgg")
import matplotlib.pyplot as plt
import tensorflow as tf
import zhusuan as zs


def gaussian(observed, n_x, stdev, n_particles):
    with zs.BayesianNet(observed=observed) as model:
        x_mean = tf.zeros([n_x])
        x = zs.Normal('x', x_mean, std=stdev, n_samples=n_particles,
                      group_ndims=1)
    return model


if __name__ == "__main__":
    # tf.set_random_seed(1)

    # Define model parameters
    n_x = 1
    # n_x = 10
    stdev = 1 / (np.arange(n_x, dtype=np.float32) + 1)

    # Define HMC parameters
    kernel_width = 0.1
    n_chains = 1000
    n_iters = 3000
    burnin = n_iters * 2 // 3

    # Build the computation graph
    def log_joint(observed):
        model = gaussian(observed, n_x, stdev, n_chains)
        return model.local_log_prob('x')

    sgmcmc = zs.SGLD(initial_step_size=0.1, final_step_size=0.001, gamma=0.55,
                     n_iters_to_final=burnin)
    x = tf.Variable(tf.zeros([n_chains, n_x]), trainable=False, name='x')
    sample_op, new_samples = sgmcmc.sample(log_joint, {}, {'x': x})

    # Run the inference
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        samples = []
        print('Sampling...')
        for i in range(n_iters):
            _, x_sample = sess.run([sample_op, new_samples['x']])
            if i >= burnin and i % 10 == 0:
                samples.append(x_sample)
        print('Finished.')
        samples = np.vstack(samples)

    # Check & plot the results
    print('Expected mean = {}'.format(np.zeros(n_x)))
    print('Sample mean = {}'.format(np.mean(samples, 0)))
    print('Expected stdev = {}'.format(stdev))
    print('Sample stdev = {}'.format(np.std(samples, 0)))
    print('Relative error of stdev = {}'.format(
        (np.std(samples, 0) - stdev) / stdev))

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
        plt.show()
