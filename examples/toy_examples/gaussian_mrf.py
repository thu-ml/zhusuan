#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import tensorflow as tf
import zhusuan as zs

if __name__ == "__main__":
    tf.set_random_seed(102)

    # Define HMC parameters
    kernel_width = 0.1
    n_chains = 1000
    n_iters = 200
    burnin = n_iters // 2
    n_leapfrogs = 5

    # Define model parameters
    n_x = 2
    x_covariance = np.array([[ 1., .5 ],
                             [ .5, 2. ]], dtype=np.float32)
    x_mean = np.array([ 1., -1. ], dtype=np.float32)
    x_precison = np.linalg.inv(x_covariance)
    x_eta = np.matmul(x_mean, x_precison)

    def log_joint(observed):
        # potentials of Gaussian Markov random field
        x = observed['x']
        edge_potentials = -0.5 * tf.matmul(x, x_precison) * x
        node_potentials = x_eta * x
        return tf.reduce_sum(edge_potentials + node_potentials, axis=-1)

    adapt_step_size = tf.placeholder(tf.bool, shape=[], name="adapt_step_size")
    adapt_mass = tf.placeholder(tf.bool, shape=[], name="adapt_mass")
    hmc = zs.HMC(step_size=1e-3, n_leapfrogs=n_leapfrogs,
                 adapt_step_size=adapt_step_size, adapt_mass=adapt_mass,
                 target_acceptance_rate=0.9)
    x = tf.Variable(tf.zeros([n_chains, n_x]), trainable=False, name='x')
    sample_op, hmc_info = hmc.sample(log_joint, {}, {'x': x})

    # Run the inference
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        samples = []
        print('Sampling...')
        for i in range(n_iters):
            _, x_sample, acc, ss = sess.run(
                [sample_op, hmc_info.samples['x'], hmc_info.acceptance_rate,
                 hmc_info.updated_step_size],
                feed_dict={adapt_step_size: i < burnin // 2,
                           adapt_mass: i < burnin // 2})
            print('Sample {}: Acceptance rate = {}, updated step size = {}'
                  .format(i, np.mean(acc), ss))
            if i >= burnin:
                samples.append(x_sample)
        print('Finished.')
        samples = np.vstack(samples)

    # Check & plot the results
    print('Expected mean:\n{}'.format(x_mean))
    print('Sample mean:\n{}'.format(np.mean(samples, 0)))
    print('Expected covariance:\n{}'.format(x_covariance))
    print('Sample covariance:\n{}'.format(np.cov(np.transpose(samples))))

    if n_x == 2:
        print('Drawing density...')
        xmin, xmax = -2, 4
        ymin, ymax = -4, 2

        # Peform the kernel density estimate
        xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        kernel = stats.gaussian_kde(np.transpose(samples))
        f = np.reshape(kernel(positions).T, xx.shape)

        fig = plt.figure()
        ax = fig.gca()
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        # Contourf plot
        cfset = ax.contourf(xx, yy, f, cmap='Blues')
        # Contour plot
        cset = ax.contour(xx, yy, f, colors='k')
        ## Label plot
        ax.clabel(cset, inline=1, fontsize=10)
        ax.set_xlabel('X0')
        ax.set_ylabel('X1')

        plt.show()
