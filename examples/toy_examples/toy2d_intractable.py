#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from six.moves import range
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import zhusuan as zs


@zs.meta_bayesian_net(scope="model")
def build_toy2d_intractable(n_particles):
    bn = zs.BayesianNet()
    z2 = bn.normal('z2', 0., std=1.35, n_samples=n_particles)
    bn.normal('z1', 0., logstd=z2)
    return bn


@zs.reuse_variables(scope="variational")
def build_mean_field_variational(n_particles):
    bn = zs.BayesianNet()
    for name in ["z1", "z2"]:
        z_mean = bn.output(name + "_mean", tf.Variable(-2.))
        z_logstd = bn.output(name + "_logstd", tf.Variable(-5.))
        bn.normal(name, z_mean, logstd=z_logstd, n_samples=n_particles)
    return bn


if __name__ == "__main__":
    # Build the computation graph
    n_particles = tf.placeholder(tf.int32, shape=[])

    model = build_toy2d_intractable(n_particles)
    variational = build_mean_field_variational(n_particles)

    lower_bound = zs.variational.elbo(
        model, {}, variational=variational, axis=0)
    cost = lower_bound.sgvb()
    optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
    infer_op = optimizer.minimize(cost)

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

    z_mean = tf.stack(variational.get(["z1_mean", "z2_mean"]))
    z_logstd = tf.stack(variational.get(["z1_logstd", "z2_logstd"]))

    # Run the inference
    iters = 1000
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for t in range(iters):
            _, lb, vmean, vlogstd = sess.run(
                [infer_op, lower_bound, z_mean, z_logstd],
                feed_dict={n_particles: 500})
            print('Iteration {}: lower bound = {}'.format(t, lb))
            draw(vmean, vlogstd)
