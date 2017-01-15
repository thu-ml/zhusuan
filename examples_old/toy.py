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
    from zhusuan.distributions_old import norm
    from zhusuan.layers_old import *
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

        :param latent: A dictionary of pairs: (string, Tensor). Each of the
            Tensor has shape (batch_size, n_samples, n_latent).
        :param observed: A dictionary of pairs: (string, Tensor). Each of the
            Tensor has shape (batch_size, n_observed).

        :return: A Tensor of shape (batch_size, n_samples). The joint log
            likelihoods.
        """
        z = latent['z']

        mu, logvar = z[:, :, 0], z[:, :, 1]
        return norm.logpdf(logvar, 0, 2.7) + norm.logpdf(
            mu, 0, tf.exp(0.5 * logvar))


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

    def draw(vmean, vlogvar):
        from scipy import stats
        plt.cla()
        xlimits = [-2, 2]
        ylimits = [-8, 4]

        def log_prob(z, x):
            mu, logvar = z[:, 0], z[:, 1]
            return stats.norm.logpdf(logvar, 0, 2.7) + stats.norm.logpdf(
                mu, 0, np.exp(0.5 * logvar))

        plot_isocontours(ax, lambda z: np.exp(log_prob(z, None)),
                         xlimits, ylimits)

        def variational_contour(z):
            return stats.multivariate_normal.pdf(
                z, vmean[0], np.diag(np.exp(vlogvar[0])))

        plot_isocontours(ax, variational_contour, xlimits, ylimits)
        plt.draw()
        plt.pause(1.0 / 30.0)

    # Build the computation graph
    model = ToyIntractablePosterior()
    n_samples = tf.placeholder(tf.int32, shape=())
    optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
    z_mean = tf.Variable(np.array([[[-2, -2]]], dtype='float32'))
    z_logvar = tf.Variable(np.array([[[-10, -10]]], dtype='float32'))
    lz_mean = InputLayer((None, 1, 2), input=z_mean)
    lz_logvar = InputLayer((None, 1, 2), input=z_logvar)
    lz = Normal([lz_mean, lz_logvar], n_samples)
    z_outputs = get_output(lz)
    latent = {'z': z_outputs}
    lower_bound = tf.reduce_mean(advi(model, {}, latent, reduction_indices=1))
    infer = optimizer.minimize(-lower_bound)
    init = tf.global_variables_initializer()

    # Run the inference
    iters = 1000
    with tf.Session() as sess:
        sess.run(init)
        for t in range(iters):
            _, lb, vmean, vlogvar = sess.run(
                [infer, lower_bound, z_mean, z_logvar],
                feed_dict={n_samples: 200})
            print('Iteration {}: lower bound = {}'.format(t, lb))
            draw(vmean[0], vlogvar[0])
