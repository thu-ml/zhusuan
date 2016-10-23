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
            for shape in [(1,), (3, 5), (1, 10, 2)]:
                samples = sess.run(norm.rvs(shape=shape))
                assert(samples.shape == shape)

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
            _ = norm.logpdf(x, mu, std)


class TestBernoulli:
    def test_rvs(self):
        with tf.Session() as sess:
            with pytest.raises(NotImplementedError):
                sess.run(bernoulli.rvs(0.1, shape=1.))

    def test_logpdf(self):
        with tf.Session() as sess:
            x = [[1, 0], [1, 1]]
            p = 0.3
            test_values = sess.run(bernoulli.logpdf(x, p))
            true_values = stats.bernoulli.logpmf(x, p)
            assert(np.abs(test_values - true_values).max() < 1e-6)

            x = [0, 1]
            p = [0, 1]
            test_values = sess.run(bernoulli.logpdf(x, p))
            true_values = stats.bernoulli.logpmf(x, p)
            assert(np.abs(test_values - true_values).max() < 1e-5)

            x = [[1, 1], [0, 1]]
            p = [0.1, 0.2, 0.3]

        with pytest.raises(ValueError):
            _ = bernoulli.logpdf(x, p)


class TestDiscrete:
    def test_rvs(self):
        p = tf.placeholder(tf.float32, shape=(None, 5, 3))
        output = discrete.rvs(p)
        assert(output.get_shape().as_list() == [None, 5, 3])

        p = np.array([[[0.5, 7., 1.], [0.6, 2., 5.]]])
        output = discrete.rvs(p)
        assert(output.get_shape().as_list() == [1, 2, 3])
        with tf.Session() as sess:
            test_values = sess.run(output)
            assert(test_values.shape == p.shape)
            p = tf.ones((3, 5))
            test_values = sess.run(discrete.rvs(p))
            assert(test_values.shape == (3, 5))

        with pytest.raises(ValueError):
            _ = discrete.rvs(np.ones(3))

    def test_logpdf(self):
        with tf.Session() as sess:
            p = np.array([[[0.5, 7., 1.], [0.5, 7., 1.]]])
            x = np.array([[[0, 1, 0], [0, 1, 0]]])
            test_values = sess.run(discrete.logpdf(x, p))[0]
            true_values = np.sum(x * np.log(p / p.sum(axis=-1, keepdims=True)),
                                 axis=-1)
            assert(np.abs(test_values - true_values).max() < 1e-6)

        with pytest.raises(ValueError):
            _ = discrete.logpdf([0, 1, 0], np.ones(3))
