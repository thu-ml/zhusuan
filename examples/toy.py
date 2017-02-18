#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import sys
import os

from six.moves import range
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import zhusuan as zs


def toy_2d_intractable_posterior(observed, n_particles):
    with zs.StochasticGraph(observed=observed) as model:
        z2_mean = tf.zeros([n_particles])
        z2_logstd = tf.ones([n_particles]) * tf.log(1.35)
        z2 = zs.Normal('z2', z2_mean, z2_logstd)
        z1_mean = tf.zeros([n_particles])
        z1 = zs.Normal('z1', z1_mean, z2)
    return model


def mean_field_variational(n_particles):
    with zs.StochasticGraph() as variational:
        z_mean = tf.Variable(np.array([-2, -2], dtype='float32'))
        z_logstd = tf.Variable(np.array([-5, -5], dtype='float32'))
        z = zs.Normal('z', z_mean, z_logstd, sample_dim=0,
                      n_samples=n_particles)
    return variational, z_mean, z_logstd


if __name__ == "__main__":
    # Build the computation graph
    n_particles = tf.placeholder(tf.int32, shape=[])

    def log_joint(observed):
        z1, z2 = tf.unstack(observed['z'], axis=1)
        model = toy_2d_intractable_posterior({'z1': z1, 'z2': z2}, n_particles)
        log_pz1, log_pz2 = model.local_log_prob(['z1', 'z2'])
        return log_pz1 + log_pz2

    variational, z_mean, z_logstd = mean_field_variational(n_particles)
    qz_samples, log_qz = variational.query('z', outputs=True,
                                           local_log_prob=True)
    log_qz = tf.reduce_sum(log_qz, -1)
    lower_bound = zs.advi(log_joint, {}, {'z': [qz_samples, log_qz]})
    optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
    infer = optimizer.minimize(-lower_bound)

    # Set up figure.
    fig = plt.figure(figsize=(8, 8), facecolor='white')
    ax = fig.add_subplot(111, frameon=False)
    plt.ion()
    plt.show(block=False)

    # Set up plotting code.
    def plot_isocontours(ax, func, xlimits, ylimits, numticks=101):
        x = np.linspace(*xlimits, num=numticks)
        y = np.linspace(*ylimits, num=numticks)
        xx, yy = np.meshgrid(x, y)
        z = func(np.concatenate([np.atleast_2d(xx.ravel()),
                                 np.atleast_2d(yy.ravel())]).T)
        z = z.reshape(xx.shape)
        ax.contour(xx, yy, z)

    def draw(vmean, vlogstd):
        from scipy import stats
        plt.cla()
        xlimits = [-2, 2]
        ylimits = [-4, 2]

        def log_prob(z):
            z1, z2 = z[:, 0], z[:, 1]
            return stats.norm.logpdf(z2, 0, 1.35) + \
                   stats.norm.logpdf(z1, 0, np.exp(z2))

        plot_isocontours(ax, lambda z: np.exp(log_prob(z)), xlimits, ylimits)

        def variational_contour(z):
            return stats.multivariate_normal.pdf(
                z, vmean, np.diag(np.exp(vlogstd)))

        plot_isocontours(ax, variational_contour, xlimits, ylimits)
        plt.draw()
        plt.pause(1.0 / 30.0)

    # Run the inference
    iters = 1000
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for t in range(iters):
            _, lb, vmean, vlogstd = sess.run(
                [infer, lower_bound, z_mean, z_logstd],
                feed_dict={n_particles: 500})
            print('Iteration {}: lower bound = {}'.format(t, lb))
            draw(vmean, vlogstd)
