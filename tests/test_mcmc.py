#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
from scipy import stats
import numpy as np
import six
import zhusuan as zs


def sample_error_with(sampler, sess, n_chains=1, n_iters=80000, thinning=50,
                      burnin=None, dtype=tf.float32):
    """
    Fig 1 in the SGHMC paper by Chen et al.
    """
    if burnin is None:
        burnin = n_iters * 2 // 3

    # Build the computation graph
    def log_joint(observed):
        x = observed['x']
        lh_noise = tf.random_normal(shape=tf.shape(x), stddev=2, dtype=dtype)
        return 2 * (x**2) - x**4 + lh_noise

    x = tf.Variable(
        tf.zeros([n_chains], dtype=dtype),
        trainable=False,
        name='x', dtype=dtype)
    sample_op, _ = sampler.sample(log_joint, {}, {'x': x})

    # Run the inference
    sess.run(tf.global_variables_initializer())
    samples = []
    for t in range(n_iters):
        _, x_sample = sess.run([sample_op, x])
        if np.isnan(x_sample.sum()):
            raise ValueError('nan encountered')
        if t >= burnin and t % thinning == 0:
            samples.append(x_sample)
    samples = np.array(samples)
    samples = samples.reshape(-1)
    A = 3
    xs = np.linspace(-A, A, 1000)
    pdfs = np.exp(2*(xs**2) - xs**4)
    pdfs = pdfs / pdfs.mean() / A / 2
    est_pdfs = stats.gaussian_kde(samples)(xs)
    return np.abs(est_pdfs - pdfs).mean()


class TestMCMC(tf.test.TestCase):

    def test_hmc(self):
        # Chen et al used n_iters=80000, n_leapfrogs=50 and n_chains=1
        # we verified visually that their configuration works, and use a small-
        # scale config here to save time
        sampler = zs.HMC(step_size=0.01, n_leapfrogs=10)
        with self.session() as sess:
            e = sample_error_with(sampler, sess, n_chains=100, n_iters=1000)
            self.assertLessEqual(e, 0.030)
