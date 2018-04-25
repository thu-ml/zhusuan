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


if __name__ == "__main__":
    tf.set_random_seed(1)

    # Define model parameters
    stdev = 0.5
    mu1 = -1
    mu2 = 3

    # Define HMC parameters
    kernel_width = 0.1
    n_chains = 1000
    n_iters = 30000
    burnin = n_iters * 2 // 3

    # Build the computation graph
    def log_joint(observed):
        x = observed['x']
        # return tf.log(tf.exp(-0.5*((x-mu1)/stdev)**2)+tf.exp(-0.5*((x-mu2)/stdev)**2))
        a1 = -0.5*((x-mu1)/stdev)**2
        a2 = -0.5*((x-mu2)/stdev)**2
        amax = tf.maximum(a1, a2)
        return amax + tf.log(tf.exp(a1-amax)+tf.exp(a2-amax))

    learning_rate = tf.placeholder(tf.float32, shape=[], name="learning_rate")
    sgmcmc = zs.SGHMC(learning_rate=learning_rate, friction=0.1, n_iter_resample_v=100000, variance_estimate=-0.1)
    x = tf.Variable(tf.zeros([n_chains]), trainable=False, name='x')
    # x = tf.Variable(tf.random_uniform([n_chains])*10-5)
    sample_op, new_samples = sgmcmc.sample(log_joint, {}, {'x': x})

    # Run the inference
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        samples = []
        print('Sampling...')
        for t in range(n_iters):
            _, x_sample = sess.run([sample_op, new_samples['x']],
                                   feed_dict={learning_rate: 0.1})
            if t >= burnin and t % 100 == 0:
                samples.append(x_sample)
        print('Finished.')
        samples = np.array(samples)
        samples = samples.reshape(-1)

    # Check & plot the results
    print('Expected mean = {}'.format(0.5*(mu1+mu2)))
    print('Sample mean = {}'.format(np.mean(samples)))
    total_stdev = np.sqrt(stdev**2+0.5*(mu1**2+mu2**2)-(0.5*(mu1+mu2))**2)
    print('Expected stdev = {}'.format(total_stdev))
    print('Sample stdev = {}'.format(np.std(samples)))
    print('Relative error of stdev = {}'.format(
        (np.std(samples) - total_stdev) / total_stdev))

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

    xs = np.linspace(-5, 5, 1000)
    ys = kde(xs, samples, n_chains)
    f, ax = plt.subplots()
    ax.plot(xs, ys)
    ax.plot(xs, 0.5*stats.norm.pdf(xs, loc=mu1, scale=stdev)+0.5*stats.norm.pdf(xs, loc=mu2, scale=stdev))
    plt.show()

# learning_rate=0.01, do not resample: when friction=0., a strange result. when
# add friction, biased to the -1 mode.