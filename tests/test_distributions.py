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
            with pytest.raises(ValueError):
                sess.run(norm.rvs(mean=tf.ones([3, 2]),
                                  logstd=tf.zeros([1, 3, 2])))
            with pytest.raises(tf.errors.InvalidArgumentError):
                mean = tf.placeholder(tf.float32, [None, 2])
                logstd = tf.placeholder(tf.float32, [None, 2])
                sess.run(norm.rvs(mean, logstd),
                         feed_dict={mean: np.ones([3, 2]),
                                    logstd: np.ones([4, 2])})
            with pytest.raises(tf.errors.InvalidArgumentError):
                sess.run(norm.rvs(np.ones([1, 2]), np.ones([1, 2]) * np.inf))

            with pytest.raises(tf.errors.InvalidArgumentError):
                sess.run(norm.rvs(np.ones([1, 2]), np.ones([1, 2]),
                                  sample_dim=None, n_samples=2))

            test_values = sess.run(
                norm.rvs(mean, logstd, sample_dim=0, n_samples=3),
                feed_dict={mean: np.ones([1, 2]), logstd: np.ones([1, 2])})
            assert(test_values.shape == (3, 1, 2))

            with pytest.raises(tf.errors.InvalidArgumentError):
                sess.run(norm.rvs(np.ones([1, 2]), np.ones([1, 2]),
                                  sample_dim=-1, n_samples=1))

    def test_logpdf(self):
        mean = tf.placeholder(tf.float32, [None, 2])
        logstd = tf.placeholder(tf.float32, [None, 2])
        x = tf.placeholder(tf.float32, [None, 10, 2])
        with tf.Session() as sess:
            logpdf = sess.run(norm.logpdf(x, mean, logstd, sample_dim=1),
                              feed_dict={mean: np.zeros([3, 2]),
                                         logstd: np.zeros([3, 2]),
                                         x: np.random.random((3, 10, 2))})
            assert(logpdf.shape == (3, 10, 2))
        mean = tf.zeros([], tf.float32)
        logstd = tf.zeros([], tf.float32)
        x = tf.zeros([], tf.float32)
        with tf.Session() as sess:
            logpdf = sess.run(norm.logpdf(x, mean, logstd))
            assert np.abs(logpdf - (-0.5 * np.log(2 * np.pi))) < 1e-7


class TestLogistic:
    def test_rvs(self):
        with pytest.raises(NotImplementedError):
            with tf.Session() as sess:
                sess.run(logistic.rvs())

    def test_cdf(self):
        mean = tf.placeholder(tf.float32, [None, 2])
        logstd = tf.placeholder(tf.float32, [None, 2])
        x = tf.placeholder(tf.float32, [None, 2, 10])
        with tf.Session() as sess:
            cdf = sess.run(logistic.cdf(x, mean, logstd, sample_dim=2),
                           feed_dict={mean: np.ones([3, 2]),
                                      logstd: np.ones([3, 2]),
                                      x: np.ones([3, 2, 10])})
            assert cdf.shape == (3, 2, 10)
            with pytest.raises(tf.errors.InvalidArgumentError):
                sess.run(logistic.cdf(x, mean, logstd, sample_dim=-1),
                         feed_dict={mean: np.ones([3, 2]),
                                    logstd: np.ones([3, 2]),
                                    x: np.ones([3, 2, 10])})
            with pytest.raises(tf.errors.InvalidArgumentError):
                sess.run(logistic.cdf(x, mean, logstd, sample_dim=-1),
                         feed_dict={mean: np.ones([3, 2]),
                                    logstd: np.ones([3, 2]) * np.inf,
                                    x: np.ones([3, 2, 10])})

class TestBernoulli:
    def test_rvs(self):
        pass

    def test_logpmf(self):
        pass


class TestDiscrete:
    def test_rvs(self):
        pass

    def test_logpmf(self):
        pass






