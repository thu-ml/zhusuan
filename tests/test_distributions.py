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
    def test_rvs_check_shape(self):
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
                sess.run(norm.rvs(np.ones([1, 2]), np.ones([1, 2]),
                                  sample_dim=None, n_samples=2))

    def test_rvs_check_numerics(self):
        with tf.Session() as sess:
            sess.run(norm.rvs(np.ones([1, 2]), np.ones([1, 2]) * np.inf,
                              check_numerics=False))
            with pytest.raises(tf.errors.InvalidArgumentError):
                sess.run(norm.rvs(np.ones([1, 2]), np.ones([1, 2]) * np.inf))

    def test_rvs_shape(self):
        with tf.Session() as sess:
            mean = tf.placeholder(tf.float32, [None, 2])
            logstd = tf.placeholder(tf.float32, [None, 2])
            a = sess.run(
                norm.rvs(mean, logstd, sample_dim=0, n_samples=3),
                feed_dict={mean: np.ones([1, 2]), logstd: np.ones([1, 2])})
            assert a.shape == (3, 1, 2)

            b = sess.run(
                norm.rvs(mean, logstd),
                feed_dict={mean: np.ones([1, 2]), logstd: np.ones([1, 2])}
            )
            assert b.shape == (1, 2)

            with pytest.raises(tf.errors.InvalidArgumentError):
                sess.run(norm.rvs(np.ones([1, 2]), np.ones([1, 2]),
                                  sample_dim=-1, n_samples=1))

    def test_rvs_static_shape(self):
        pass

    def test_rvs_reparameterized(self):
        with tf.Session() as sess:
            mean = tf.ones([10, 10])
            logstd = tf.zeros([10, 10])
            a = norm.rvs(mean, logstd, reparameterized=True)
            mean_grads, logstd_grads = tf.gradients(tf.reduce_mean(a),
                                                    [mean, logstd])
            mean_grads_, logstd_grads_ = sess.run([mean_grads, logstd_grads])
            assert np.abs(mean_grads_).max() > 1e-6
            assert np.abs(logstd_grads_).max() > 1e-6

            b = norm.rvs(mean, logstd)
            mean_grads, logstd_grads = tf.gradients(tf.reduce_mean(b),
                                                    [mean, logstd])
            assert mean_grads is None
            assert logstd_grads is None

    def test_logpdf_static_shape(self):
        pass

    def test_logpdf_shape(self):
        with tf.Session() as sess:
            with pytest.raises(tf.errors.InvalidArgumentError):
                sess.run(norm.logpdf(np.ones([2]), np.ones([2]), np.ones([2]),
                                     sample_dim=-1))

        mean = tf.placeholder(tf.float32, [None, 2])
        logstd = tf.placeholder(tf.float32, [None, 2])
        x = tf.placeholder(tf.float32, [None, 10, 2])
        y = tf.placeholder(tf.float32, [None, 2])
        with tf.Session() as sess:
            x_logpdf = sess.run(norm.logpdf(x, mean, logstd, sample_dim=1),
                                feed_dict={mean: np.zeros([3, 2]),
                                           logstd: np.zeros([3, 2]),
                                           x: np.random.random([3, 10, 2])})
            y_logpdf = sess.run(norm.logpdf(y, mean, logstd),
                                feed_dict={mean: np.zeros([3, 2]),
                                           logstd: np.zeros([3, 2]),
                                           y: np.random.random([3, 2])})
            assert x_logpdf.shape == (3, 10, 2)
            assert y_logpdf.shape == (3, 2)

    def test_logpdf_check_numerics(self):
        mean = tf.constant([2, 3], dtype=tf.float32)
        logstd = tf.constant([4, -100], dtype=tf.float32)
        x = tf.constant([3, 3.1], dtype=tf.float32)
        with tf.Session() as sess:
            with pytest.raises(tf.errors.InvalidArgumentError):
                sess.run(norm.logpdf(x, mean, logstd))

    def test_logpdf_value(self):
        mean = tf.zeros([], tf.float32)
        logstd = tf.zeros([], tf.float32)
        x = tf.zeros([], tf.float32)
        with tf.Session() as sess:
            logpdf = sess.run(norm.logpdf(x, mean, logstd))
            assert np.abs(logpdf - (-0.5 * np.log(2 * np.pi))) < 1e-6

        mean_value = np.array([2, 3], dtype='float32')
        logstd_value = np.array([4, -10], dtype='float32')
        x_value = np.array([3, 3.1], dtype='float32')
        mean = tf.constant(mean_value)
        logstd = tf.constant(logstd_value)
        x = tf.constant(x_value)
        with tf.Session() as sess:
            test_value = sess.run(norm.logpdf(x, mean, logstd))
            true_value = stats.norm.logpdf(x_value, mean_value,
                                           np.exp(logstd_value))
            assert np.abs((test_value - true_value) / true_value).max() < 1e-6

        mean_value = np.array([2, 3], dtype='float32')
        logstd_value = np.array([4, 0], dtype='float32')
        x_value = np.array([3, 2, 3], dtype='float32')
        mean = tf.constant(mean_value)
        logstd = tf.constant(logstd_value)
        x = tf.constant(x_value)
        with tf.Session() as sess:
            test_value = sess.run(norm.logpdf(x, mean, logstd, sample_dim=1))
            true_value = stats.norm.logpdf(
                x_value, np.expand_dims(mean_value, 1),
                np.exp(np.expand_dims(logstd_value, 1)))
            assert np.abs(test_value - true_value).max() < 1e-6


class TestLogistic:
    def test_rvs(self):
        with pytest.raises(NotImplementedError):
            logistic.rvs()

    def test_cdf_shape(self):
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

    def test_cdf_static_shape(self):
        pass

    def test_cdf_check_numerics(self):
        with tf.Session() as sess:
            mean = tf.constant(np.ones([3, 2], dtype='float32'))
            logstd = tf.constant(np.ones([3, 2], dtype='float32') * (-100))
            x = tf.constant(np.ones([3, 2, 10], dtype='float32'))
            with pytest.raises(tf.errors.InvalidArgumentError):
                sess.run(logistic.cdf(x, mean, logstd, sample_dim=2))

    def test_cdf_value(self):
        mean_value = np.array([2, 3], dtype='float32')
        logstd_value = np.array([4, -10], dtype='float32')
        x_value = np.array([3, 3.1], dtype='float32')
        mean = tf.constant(mean_value)
        logstd = tf.constant(logstd_value)
        x = tf.constant(x_value)
        with tf.Session() as sess:
            test_value = sess.run(logistic.cdf(x, mean, logstd))
            true_value = stats.logistic.cdf(x_value, mean_value,
                                            np.exp(logstd_value))
            assert np.abs(
                (test_value - true_value) / true_value).max() < 1e-6

        mean_value = np.array([2, 3], dtype='float32')
        logstd_value = np.array([4, 0], dtype='float32')
        x_value = np.array([3, 2, 3], dtype='float32')
        mean = tf.constant(mean_value)
        logstd = tf.constant(logstd_value)
        x = tf.constant(x_value)
        with tf.Session() as sess:
            test_value = sess.run(
                logistic.cdf(x, mean, logstd, sample_dim=1))
            true_value = stats.logistic.cdf(
                x_value, np.expand_dims(mean_value, 1),
                np.exp(np.expand_dims(logstd_value, 1)))
            assert np.abs(test_value - true_value).max() < 1e-6


class TestBernoulli:
    def test_rvs(self):
        with tf.Session() as sess:
            p = tf.placeholder(tf.float32, [None, 2])
            a = sess.run(
                bernoulli.rvs(p, sample_dim=0, n_samples=3),
                feed_dict={p: np.ones([1, 2])})
            assert a.shape == (3, 1, 2)
            assert np.all((np.abs(a - 1) < 1e-8) | (np.abs(a) < 1e-8))

            b = sess.run(
                bernoulli.rvs(p),
                feed_dict={p: np.ones([1, 2])}
            )
            assert b.shape == (1, 2)
            assert np.all((np.abs(b - 1) < 1e-8) | (np.abs(b) < 1e-8))

            with pytest.raises(tf.errors.InvalidArgumentError):
                sess.run(bernoulli.rvs(np.ones([1, 2]),
                                       sample_dim=None, n_samples=2))

            with pytest.raises(tf.errors.InvalidArgumentError):
                sess.run(bernoulli.rvs(np.ones([1, 2]),
                                       sample_dim=-1, n_samples=1))

    def test_logpmf(self):
        pass


class TestDiscrete:
    def test_rvs(self):
        pass

    def test_logpmf(self):
        pass
