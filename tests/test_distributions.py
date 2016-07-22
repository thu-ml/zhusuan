#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

import tensorflow as tf
from scipy import stats
import numpy as np

from .context import zhusuan
from zhusuan.distributions import *


class TestNormal:
    def test_rvs(self):
        with tf.Session() as sess:
            for size in [(1,), (3, 5), (1, 10, 2)]:
                samples = sess.run(norm.rvs(size=size))
                assert(samples.shape == size)

    def test_logpdf(self):
        with tf.Session() as sess:
            x = [[-1., -2.3], [5., 0.]]
            test_values = sess.run(norm.logpdf(x))
            true_values = stats.norm.logpdf(x)
            assert(np.abs(test_values - true_values).max() < 1e-6)

            mu = [[-5., 8.], [2., 12.]]
            std = [[0.5, 3.], [4., 10.]]
            test_values = sess.run(norm.logpdf(x, mu, std))
            true_values = stats.norm.logpdf(x, mu, std)
            assert(np.abs(test_values - true_values).max() < 1e-6)

    def test_pdf(self):
        pass
