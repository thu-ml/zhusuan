#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import sys
import os

from six.moves import range
import tensorflow as tf
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from zhusuan.distributions import norm
    from zhusuan.variational import ReparameterizedNormal, advi
except:
    raise ImportError()


class ToyIntractablePosterior:
    """
    A Toy 2D intractable distribution.
    """
    def __init__(self):
        pass

    def log_prob(self, z, x):
        """
        The joint likelihood.

        :param z: Tensor of shape (batch_size, samples, n_z). n_z is the
            dimension of latent variables.
        :param x: Tensor of shape (batch_size, n_x). n_x is the dimension of
            observed variables (data).

        :return: A Tensor of shape (batch_size, samples). The joint log
            likelihoods.
        """
        mu, log_sigma = z[:, :, 0], z[:, :, 1]
        return norm.logpdf(log_sigma, 0, 1.35) + norm.logpdf(
            mu, 0, tf.exp(log_sigma))


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
        plt.contour(xx, yy, z)

    def draw(vmean, vlogstd):
        from scipy import stats
        plt.cla()
        xlimits = [-2, 2]
        ylimits = [-4, 2]

        def log_prob(z, x):
            mu, log_sigma = z[:, 0], z[:, 1]
            return stats.norm.logpdf(log_sigma, 0, 1.35) + stats.norm.logpdf(
                mu, 0, np.exp(log_sigma))

        plot_isocontours(ax, lambda z: np.exp(log_prob(z, None)),
                         xlimits, ylimits)

        def variational_contour(z):
            return stats.multivariate_normal.pdf(
                z, vmean[0], np.diag(np.exp(2 * vlogstd[0])))

        plot_isocontours(ax, variational_contour, xlimits, ylimits)
        plt.draw()
        plt.pause(1.0 / 30.0)

    # Build the computation graph
    model = ToyIntractablePosterior()
    optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
    vz_mean = tf.Variable(np.array([[-2, -2]], dtype='float32'))
    vz_logstd = tf.Variable(np.array([[-5, -5]], dtype='float32'))
    variational = ReparameterizedNormal(vz_mean, vz_logstd)
    grads, lower_bound = advi(model, None, variational, 200, optimizer)
    infer = optimizer.apply_gradients(grads)
    init = tf.initialize_all_variables()

    # Run the inference
    iters = 1000
    with tf.Session() as sess:
        sess.run(init)
        for t in range(iters):
            _, lb, vmean, vlogstd = sess.run([infer, lower_bound, vz_mean,
                                              vz_logstd])
            print('Iteration {}: lower bound = {}'.format(t, lb))
            draw(vmean, vlogstd)
