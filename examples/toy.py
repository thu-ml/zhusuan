#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import sys
import os

from six.moves import range
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from zhusuan.distributions import norm
    from zhusuan.model import *
    from zhusuan.variational import advi
except:
    raise ImportError()


class ToyIntractablePosterior:
    """
    A Toy 2D intractable distribution.
    """
    def __init__(self):
        pass

    def log_prob(self, latent, observed, given):
        """
        The log joint probability function.

        :param latent: A dictionary of pairs: (string, Tensor).
        :param observed: A dictionary of pairs: (string, Tensor).
        :param given: A dictionary of pairs: (string, Tensor).

        :return: A Tensor. The joint log likelihoods.
        """
        z = latent['z']

        mu, logstd = z[:, 0], z[:, 1]
        return norm.logpdf(logstd) + norm.logpdf(mu, 0, logstd)


if __name__ == "__main__":
    # Set up figure.
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(8, 8), facecolor='white')
    ax = fig.add_subplot(111, frameon=False)
    plt.ion()
    plt.show(block=False)

    # Set up plotting code.
    def plot_isocontours(ax, func, xlimits, ylimits, numticks=101):
        x = np.linspace(*xlimits, num=numticks)
        y = np.linspace(*ylimits, num=numticks)
        xx, yy = np.meshgrid(x, y)
        zs = func(np.concatenate(
            [np.atleast_2d(xx.ravel()), np.atleast_2d(yy.ravel())]).T)
        z = zs.reshape(xx.shape)
        ax.contour(xx, yy, z)

    def draw(vmean, vlogstd):
        from scipy import stats
        plt.cla()
        xlimits = [-2, 2]
        ylimits = [-4, 2]

        def log_prob(z, x):
            mu, logstd = z[:, 0], z[:, 1]
            return stats.norm.logpdf(logstd, 0, 1.35) + \
                stats.norm.logpdf(mu, 0, np.exp(logstd))

        plot_isocontours(ax, lambda z: np.exp(log_prob(z, None)),
                         xlimits, ylimits)

        def variational_contour(z):
            return stats.multivariate_normal.pdf(
                z, vmean, np.diag(np.exp(vlogstd)))

        plot_isocontours(ax, variational_contour, xlimits, ylimits)
        plt.draw()
        plt.pause(1.0 / 30.0)

    # Build the computation graph
    model = ToyIntractablePosterior()
    n_particles = tf.placeholder(tf.int32, shape=[])
    optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
    with StochasticGraph() as variational:
        z_mean = tf.Variable(np.array([-2, -2], dtype='float32'))
        z_logstd = tf.Variable(np.array([-5, -5], dtype='float32'))
        qz = Normal(z_mean, z_logstd, sample_dim=0, n_samples=n_particles)
    z, z_logpdf = variational.get_output(qz)
    z_logpdf = tf.reduce_sum(z_logpdf, -1)
    lower_bound = tf.reduce_mean(advi(
        model, {}, {'z': [z, z_logpdf]}, reduction_indices=0))
    infer = optimizer.minimize(-lower_bound)
    init = tf.initialize_all_variables()

    # Run the inference
    iters = 1000
    with tf.Session() as sess:
        sess.run(init)
        for t in range(iters):
            _, lb, vmean, vlogstd = sess.run(
                [infer, lower_bound, z_mean, z_logstd],
                feed_dict={n_particles: 200})
            print('Iteration {}: lower bound = {}'.format(t, lb))
            draw(vmean, vlogstd)
