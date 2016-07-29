#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
from scipy import stats
import numpy as np
import pytest

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
            test_values = sess.run(norm.logpdf(x, tf.constant(mu), std))
            true_values = stats.norm.logpdf(x, mu, std)
            assert(np.abs(test_values - true_values).max() < 1e-6)

            mu = [0.1, 0.2, 0.3]
            std = np.ones(3)
            with pytest.raises(ValueError):
                sess.run(norm.logpdf(x, mu, std))


class TestBernoulli:
    def test_rvs(self):
        with tf.Session() as sess:
            with pytest.raises(NotImplementedError):
                sess.run(bernoulli.rvs(0.1, size=1.))

    def test_logpdf(self):
        with tf.Session() as sess:
            x = [[1, 0], [1, 1]]
            p = 0.3
            test_values = sess.run(bernoulli.logpdf(tf.constant(x), p))
            true_values = stats.bernoulli.logpmf(x, p)
            assert(np.abs(test_values - true_values).max() < 1e-6)

            x = [0, 1]
            p = [0, 1]
            test_values = sess.run(bernoulli.logpdf(x, p))
            print(test_values)
            true_values = stats.bernoulli.logpmf(x, p)
            print(true_values)
            assert(np.abs(test_values - true_values).max() < 1e-5)

            x = [[1, 1], [0, 1]]
            p = [0.1, 0.2, 0.3]
            with pytest.raises(ValueError):
                sess.run(bernoulli.logpdf(x, p))
