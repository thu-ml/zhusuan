#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import six.moves as sm
import tensorflow as tf
import numpy as np

from zhusuan.distributions import norm


class ToyIntractablePosterior:
    """
    A Toy 2D intractable distribution.
    """
    def __init__(self):
        pass

    def log_prob(self, z, x):
        mu, log_sigma = z[:, 0], z[:, 1]
        return norm.logpdf(log_sigma, 0, 1.35) + norm.logpdf(
            mu, 0, tf.exp(log_sigma))


def advi(model, x, vz, n_samples=1, optimizer=tf.train.AdamOptimizer()):
    """
    Implements the automatic differentiation variational inference (ADVI)
    algorithm. For now we assume all latent variables have been transformed in
    the model definition to have support on R^n.

    :param model: An model object that has a method logprob(z, x) to compute
        joint likelihood of the model.
    :param x: 2-D Tensor. Observed data.
    :param vz: A Tensorflow node that has shape (a1, 2), where (a1,) is the
        shape of latent variables z, and the last dimension represents mean
        and log standard deviation of the variational posterior.
    :param n_samples: (Optional) int. Number of posterior samples used to
        estimate the gradients.
    :param optimizer: (Optional) Tensorflow optimizer object. Default to be
        AdamOptimizer.

    :return: A Tensorflow computation graph of the inference procedure.
    :return: A 0-D Tensor. The variational lower bound.
    """
    vz_mean, vz_logstd = vz[:, 0], vz[:, 1]
    samples = norm.rvs(size=(n_samples, tf.shape(vz)[0])) * tf.exp(vz_logstd) \
        + vz_mean
    lower_bound = model.log_prob(samples, x) - tf.reduce_sum(norm.logpdf(
        samples, vz_mean, tf.exp(vz_logstd)), 1)
    lower_bound = tf.reduce_mean(lower_bound)
    return optimizer.minimize(-lower_bound), lower_bound


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

    def draw(vz):
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
        vz_mean, vz_logstd = vz[:, 0], vz[:, 1]

        def variational_contour(z):
            return stats.multivariate_normal.pdf(z, vz_mean,
                                                 np.diag(np.exp(2 * vz_logstd)))

        plot_isocontours(ax, variational_contour, xlimits, ylimits)
        plt.draw()
        plt.pause(1.0 / 30.0)

    model = ToyIntractablePosterior()
    optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
    vz = tf.Variable(np.array([[-2, -5],
                               [-2, -5]], dtype='float32'))
    infer, lb = advi(model, None, vz, 200, optimizer)
    init = tf.initialize_all_variables()

    iters = 1000
    with tf.Session() as sess:
        sess.run(init)
        for t in sm.xrange(iters):
            _, lower_bound = sess.run([infer, lb])
            print('Iteration {}: lower bound = {}'.format(t, lower_bound))
            draw(vz.eval())
