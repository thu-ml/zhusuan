#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
from scipy import stats, misc
import numpy as np
import pytest

from .context import zhusuan
from zhusuan.distributions_old import *


class TestUniform:
    def test_rvs_check_shape(self):
        with tf.Session() as sess:
            with pytest.raises(ValueError):
                sess.run(uniform.rvs(minval=tf.zeros([3, 2]),
                                     maxval=tf.ones([1, 3, 2])))

            with pytest.raises(tf.errors.InvalidArgumentError):
                minval = tf.placeholder(tf.float32, [None, 2])
                maxval = tf.placeholder(tf.float32, [None, 2])
                sess.run(uniform.rvs(minval, maxval),
                         feed_dict={minval: np.zeros([3, 2]),
                                    maxval: np.ones([4, 2])})

            with pytest.raises(tf.errors.InvalidArgumentError):
                sess.run(uniform.rvs(np.zeros([1, 2]), np.ones([1, 2]),
                                     sample_dim=None, n_samples=2))

    def test_rvs_shape(self):
        with tf.Session() as sess:
            minval = tf.placeholder(tf.float32, [None, 2])
            maxval = tf.placeholder(tf.float32, [None, 2])
            a = sess.run(
                uniform.rvs(minval, maxval, sample_dim=0, n_samples=3),
                feed_dict={minval: np.zeros([1, 2]), maxval: np.ones([1, 2])})
            assert a.shape == (3, 1, 2)

            b = sess.run(
                uniform.rvs(minval, maxval),
                feed_dict={minval: np.zeros([1, 2]), maxval: np.ones([1, 2])})
            assert b.shape == (1, 2)

            sample_dim = tf.placeholder(tf.int32, shape=())
            with pytest.raises(tf.errors.InvalidArgumentError):
                sess.run(uniform.rvs(np.zeros([1, 2]), np.ones([1, 2]),
                                     sample_dim=sample_dim, n_samples=1),
                         feed_dict={sample_dim: -1})

    def test_rvs_static_shape(self):
        minval = tf.zeros([2, 3])
        maxval = tf.ones([2, 3])
        a = uniform.rvs(minval, maxval)
        assert a.get_shape().as_list() == [2, 3]
        with tf.Session() as sess:
            assert sess.run(a).shape == (2, 3)

        b = uniform.rvs(minval, maxval, sample_dim=0, n_samples=4)
        assert b.get_shape().as_list() == [4, 2, 3]
        with tf.Session() as sess:
            assert sess.run(b).shape == (4, 2, 3)

        c = uniform.rvs(minval, maxval, sample_dim=1, n_samples=4)
        assert c.get_shape().as_list() == [2, 4, 3]
        with tf.Session() as sess:
            assert sess.run(c).shape == (2, 4, 3)

        d = uniform.rvs(minval, maxval, sample_dim=2, n_samples=4)
        assert d.get_shape().as_list() == [2, 3, 4]
        with tf.Session() as sess:
            assert sess.run(d).shape == (2, 3, 4)

        sample_dim = tf.placeholder(tf.int32, shape=())
        e = uniform.rvs(minval, maxval, sample_dim=sample_dim, n_samples=2)
        assert e.get_shape().as_list() == [None, None, None]

        with pytest.raises(ValueError):
            uniform.rvs(minval, maxval, sample_dim=-1, n_samples=4)

    def test_rvs_reparameterized(self):
        with tf.Session() as sess:
            minval = tf.zeros([10, 10])
            maxval = tf.ones([10, 10])
            a = uniform.rvs(minval, maxval, reparameterized=True)
            minval_grads, maxval_grads = tf.gradients(tf.reduce_mean(a),
                                                      [minval, maxval])
            minval_grads_, maxval_grads_ = sess.run([minval_grads,
                                                     maxval_grads])
            assert np.abs(minval_grads_).max() > 1e-6
            assert np.abs(maxval_grads_).max() > 1e-6

            b = uniform.rvs(minval, maxval)
            minval_grads, maxval_grads = tf.gradients(tf.reduce_mean(b),
                                                      [minval, maxval])
            assert minval_grads is None
            assert maxval_grads is None

    def test_logpdf_static_shape(self):
        minval = tf.ones([2, 3])
        maxval = tf.zeros([2, 3])
        a = tf.zeros([2, 3])
        a_logpdf = uniform.logpdf(a, minval, maxval)
        assert a_logpdf.get_shape().as_list() == [2, 3]
        with tf.Session() as sess:
            assert sess.run(a_logpdf).shape == (2, 3)

        b = tf.ones([4, 2, 3])
        b_logpdf = uniform.logpdf(b, minval, maxval, sample_dim=0)
        assert b_logpdf.get_shape().as_list() == [4, 2, 3]
        with tf.Session() as sess:
            assert sess.run(b_logpdf).shape == (4, 2, 3)

        c = tf.zeros([2, 4, 3])
        c_logpdf = uniform.logpdf(c, minval, maxval, sample_dim=1)
        assert c_logpdf.get_shape().as_list() == [2, 4, 3]
        with tf.Session() as sess:
            assert sess.run(c_logpdf).shape == (2, 4, 3)

        d = tf.zeros([2, 3, 4])
        d_logpdf = uniform.logpdf(d, minval, maxval, sample_dim=2)
        assert d_logpdf.get_shape().as_list() == [2, 3, 4]
        with tf.Session() as sess:
            assert sess.run(d_logpdf).shape == (2, 3, 4)

        with pytest.raises(ValueError):
            uniform.logpdf(d, minval, maxval, sample_dim=-1)

    def test_logpdf_shape(self):
        with tf.Session() as sess:
            sample_dim = tf.placeholder(tf.int32, shape=())
            with pytest.raises(tf.errors.InvalidArgumentError):
                sess.run(
                    uniform.logpdf(np.ones([2]), np.zeros([2]), np.ones([2]),
                                   sample_dim=sample_dim),
                    feed_dict={sample_dim: -1})

        minval = tf.placeholder(tf.float32, [None, 2])
        maxval = tf.placeholder(tf.float32, [None, 2])
        x = tf.placeholder(tf.float32, [None, 10, 2])
        y = tf.placeholder(tf.float32, [None, 2])
        with tf.Session() as sess:
            x_logpdf = sess.run(uniform.logpdf(x, minval, maxval,
                                               sample_dim=1),
                                feed_dict={minval: np.zeros([3, 2]),
                                           maxval: np.ones([3, 2]),
                                           x: np.random.random([3, 10, 2])})
            y_logpdf = sess.run(uniform.logpdf(y, minval, maxval),
                                feed_dict={minval: np.zeros([3, 2]),
                                           maxval: np.ones([3, 2]),
                                           y: np.random.random([3, 2])})
            assert x_logpdf.shape == (3, 10, 2)
            assert y_logpdf.shape == (3, 2)

    def test_logpdf_value(self):
        with tf.Session() as sess:
            assert np.abs(sess.run(uniform.logpdf(0, 0, 1)) - 0.) < 1e-6
            assert sess.run(uniform.logpdf(1, 0, 1)) == -np.inf
            assert np.all(sess.run(uniform.logpdf([-1, 2], [-3, 3], [-2, 4]))
                          == np.array([-np.inf, -np.inf]))

        minval_value = np.array([2, -10, -5], dtype='float32')
        maxval_value = np.array([4, 4, -1], dtype='float32')
        x_value = np.array([3, 3.9, -4.9], dtype='float32')
        minval = tf.constant(minval_value)
        maxval = tf.constant(maxval_value)
        x = tf.constant(x_value)
        with tf.Session() as sess:
            test_value = sess.run(uniform.logpdf(x, minval, maxval))
            true_value = stats.uniform.logpdf(x_value, minval_value,
                                              maxval_value - minval_value)
            assert np.abs((test_value - true_value) / true_value).max() < 1e-6

        minval = -np.ones([2, 3], dtype='float32') * 2
        maxval = np.ones([2, 3], dtype='float32') * 2
        x = np.random.random([2, 4, 3]).astype('float32')
        with tf.Session() as sess:
            test_value = sess.run(
                uniform.logpdf(x, minval, maxval, sample_dim=1))
            true_value = stats.uniform.logpdf(
                x, np.expand_dims(minval, 1),
                np.expand_dims(maxval - minval, 1))
            assert test_value.shape == (2, 4, 3)
            assert np.abs(test_value - true_value).max() < 1e-6


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

            sample_dim = tf.placeholder(tf.int32, shape=())
            with pytest.raises(tf.errors.InvalidArgumentError):
                sess.run(norm.rvs(np.ones([1, 2]), np.ones([1, 2]),
                                  sample_dim=sample_dim, n_samples=1),
                         feed_dict={sample_dim: -1})

    def test_rvs_static_shape(self):
        mean = tf.ones([2, 3])
        logstd = tf.zeros([2, 3])
        a = norm.rvs(mean, logstd)
        assert a.get_shape().as_list() == [2, 3]
        with tf.Session() as sess:
            assert sess.run(a).shape == (2, 3)

        b = norm.rvs(mean, logstd, sample_dim=0, n_samples=4)
        assert b.get_shape().as_list() == [4, 2, 3]
        with tf.Session() as sess:
            assert sess.run(b).shape == (4, 2, 3)

        c = norm.rvs(mean, logstd, sample_dim=1, n_samples=4)
        assert c.get_shape().as_list() == [2, 4, 3]
        with tf.Session() as sess:
            assert sess.run(c).shape == (2, 4, 3)

        d = norm.rvs(mean, logstd, sample_dim=2, n_samples=4)
        assert d.get_shape().as_list() == [2, 3, 4]
        with tf.Session() as sess:
            assert sess.run(d).shape == (2, 3, 4)

        sample_dim = tf.placeholder(tf.int32, shape=())
        e = norm.rvs(mean, logstd, sample_dim=sample_dim, n_samples=2)
        assert e.get_shape().as_list() == [None, None, None]

        with pytest.raises(ValueError):
            norm.rvs(mean, logstd, sample_dim=-1, n_samples=4)

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
        mean = tf.ones([2, 3])
        logstd = tf.zeros([2, 3])
        a = tf.ones([2, 3])
        a_logpdf = norm.logpdf(a, mean, logstd)
        assert a_logpdf.get_shape().as_list() == [2, 3]
        with tf.Session() as sess:
            assert sess.run(a_logpdf).shape == (2, 3)

        b = tf.ones([4, 2, 3])
        b_logpdf = norm.logpdf(b, mean, logstd, sample_dim=0)
        assert b_logpdf.get_shape().as_list() == [4, 2, 3]
        with tf.Session() as sess:
            assert sess.run(b_logpdf).shape == (4, 2, 3)

        c = tf.ones([2, 4, 3])
        c_logpdf = norm.logpdf(c, mean, logstd, sample_dim=1)
        assert c_logpdf.get_shape().as_list() == [2, 4, 3]
        with tf.Session() as sess:
            assert sess.run(c_logpdf).shape == (2, 4, 3)

        d = tf.ones([2, 3, 4])
        d_logpdf = norm.logpdf(d, mean, logstd, sample_dim=2)
        assert d_logpdf.get_shape().as_list() == [2, 3, 4]
        with tf.Session() as sess:
            assert sess.run(d_logpdf).shape == (2, 3, 4)

        with pytest.raises(ValueError):
            norm.logpdf(d, mean, logstd, sample_dim=-1)

    def test_logpdf_shape(self):
        with tf.Session() as sess:
            sample_dim = tf.placeholder(tf.int32, shape=())
            with pytest.raises(tf.errors.InvalidArgumentError):
                sess.run(norm.logpdf(np.ones([2]), np.ones([2]), np.ones([2]),
                                     sample_dim=sample_dim),
                         feed_dict={sample_dim: -1})

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

        mean = np.ones([2, 3], dtype='float32')
        logstd = np.ones([2, 3], dtype='float32')
        x = np.random.random([2, 4, 3]).astype('float32') * 5
        with tf.Session() as sess:
            test_value = sess.run(norm.logpdf(x, mean, logstd, sample_dim=1))
            true_value = stats.norm.logpdf(x, np.expand_dims(mean, 1),
                                           np.exp(np.expand_dims(logstd, 1)))
            assert test_value.shape == (2, 4, 3)
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

            sample_dim = tf.placeholder(tf.int32, shape=())
            with pytest.raises(tf.errors.InvalidArgumentError):
                sess.run(logistic.cdf(x, mean, logstd, sample_dim=sample_dim),
                         feed_dict={mean: np.ones([3, 2]),
                                    logstd: np.ones([3, 2]),
                                    x: np.ones([3, 2, 10]),
                                    sample_dim: -1})

    def test_cdf_static_shape(self):
        mean = tf.ones([2, 3])
        logstd = tf.zeros([2, 3])
        a = tf.ones([2, 3])
        a_logpdf = logistic.cdf(a, mean, logstd)
        assert a_logpdf.get_shape().as_list() == [2, 3]
        with tf.Session() as sess:
            assert sess.run(a_logpdf).shape == (2, 3)

        b = tf.ones([4, 2, 3])
        b_logpdf = logistic.cdf(b, mean, logstd, sample_dim=0)
        assert b_logpdf.get_shape().as_list() == [4, 2, 3]
        with tf.Session() as sess:
            assert sess.run(b_logpdf).shape == (4, 2, 3)

        c = tf.ones([2, 4, 3])
        c_logpdf = logistic.cdf(c, mean, logstd, sample_dim=1)
        assert c_logpdf.get_shape().as_list() == [2, 4, 3]
        with tf.Session() as sess:
            assert sess.run(c_logpdf).shape == (2, 4, 3)

        d = tf.ones([2, 3, 4])
        d_logpdf = logistic.cdf(d, mean, logstd, sample_dim=2)
        assert d_logpdf.get_shape().as_list() == [2, 3, 4]
        with tf.Session() as sess:
            assert sess.run(d_logpdf).shape == (2, 3, 4)

        with pytest.raises(ValueError):
            logistic.cdf(d, mean, logstd, sample_dim=-1)

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

        mean = np.ones([2, 3], dtype='float32')
        logstd = np.ones([2, 3], dtype='float32')
        x = np.random.random([2, 4, 3]).astype('float32') * 5
        with tf.Session() as sess:
            test_value = sess.run(logistic.cdf(x, mean, logstd, sample_dim=1))
            true_value = stats.logistic.cdf(x, np.expand_dims(mean, 1),
                                            np.exp(np.expand_dims(logstd, 1)))
            assert test_value.shape == (2, 4, 3)
            assert np.abs(test_value - true_value).max() < 1e-6


class TestBernoulli:
    def test_rvs_static_shape(self):
        logits = tf.zeros([2, 3])
        a = bernoulli.rvs(logits)
        assert a.get_shape().as_list() == [2, 3]
        with tf.Session() as sess:
            assert sess.run(a).shape == (2, 3)

        b = bernoulli.rvs(logits, sample_dim=0, n_samples=4)
        assert b.get_shape().as_list() == [4, 2, 3]
        with tf.Session() as sess:
            assert sess.run(b).shape == (4, 2, 3)

        c = bernoulli.rvs(logits, sample_dim=1, n_samples=4)
        assert c.get_shape().as_list() == [2, 4, 3]
        with tf.Session() as sess:
            assert sess.run(c).shape == (2, 4, 3)

        d = bernoulli.rvs(logits, sample_dim=2, n_samples=4)
        assert d.get_shape().as_list() == [2, 3, 4]
        with tf.Session() as sess:
            assert sess.run(d).shape == (2, 3, 4)

        sample_dim = tf.placeholder(tf.int32, shape=())
        e = bernoulli.rvs(logits, sample_dim=sample_dim, n_samples=2)
        assert e.get_shape().as_list() == [None, None, None]

        with pytest.raises(ValueError):
            bernoulli.rvs(logits, sample_dim=-1, n_samples=4)

    def test_rvs(self):
        with tf.Session() as sess:
            logits = tf.placeholder(tf.float32, [None, 2])
            a = sess.run(
                bernoulli.rvs(logits, sample_dim=0, n_samples=3),
                feed_dict={logits: np.ones([1, 2])})
            assert a.shape == (3, 1, 2)
            assert np.all((np.abs(a - 1) < 1e-8) | (np.abs(a) < 1e-8))

            b = sess.run(
                bernoulli.rvs(logits),
                feed_dict={logits: np.ones([1, 2])}
            )
            assert b.shape == (1, 2)
            assert np.all((np.abs(b - 1) < 1e-8) | (np.abs(b) < 1e-8))

            with pytest.raises(tf.errors.InvalidArgumentError):
                sess.run(bernoulli.rvs(np.ones([1, 2]),
                                       sample_dim=None, n_samples=2))

            sample_dim = tf.placeholder(tf.int32, shape=())
            with pytest.raises(tf.errors.InvalidArgumentError):
                sess.run(bernoulli.rvs(np.ones([1, 2]),
                                       sample_dim=sample_dim, n_samples=1),
                         feed_dict={sample_dim: -1})

    def test_logpmf_static_shape(self):
        logits = tf.ones([2, 3])
        a = tf.ones([2, 3])
        a_logpmf = bernoulli.logpmf(a, logits)
        assert a_logpmf.get_shape().as_list() == [2, 3]
        with tf.Session() as sess:
            assert sess.run(a_logpmf).shape == (2, 3)

        b = tf.ones([4, 2, 3])
        b_logpmf = bernoulli.logpmf(b, logits, sample_dim=0)
        assert b_logpmf.get_shape().as_list() == [4, 2, 3]
        with tf.Session() as sess:
            assert sess.run(b_logpmf).shape == (4, 2, 3)

        c = tf.ones([2, 4, 3])
        c_logpmf = bernoulli.logpmf(c, logits, sample_dim=1)
        assert c_logpmf.get_shape().as_list() == [2, 4, 3]
        with tf.Session() as sess:
            assert sess.run(c_logpmf).shape == (2, 4, 3)

        d = tf.ones([2, 3, 4])
        d_logpmf = bernoulli.logpmf(d, logits, sample_dim=2)
        assert d_logpmf.get_shape().as_list() == [2, 3, 4]
        with tf.Session() as sess:
            assert sess.run(d_logpmf).shape == (2, 3, 4)

        with pytest.raises(ValueError):
            bernoulli.logpmf(d, logits, sample_dim=-1)

    def test_logpmf_check_shape(self):
        with tf.Session() as sess:
            with pytest.raises(ValueError):
                sess.run(bernoulli.logpmf(tf.ones([3, 2]),
                                          tf.zeros([1, 3, 2])))

            with pytest.raises(tf.errors.InvalidArgumentError):
                x = tf.placeholder(tf.float32, [None, 2])
                logits = tf.placeholder(tf.float32, [None, 2])
                sess.run(bernoulli.logpmf(x, logits),
                         feed_dict={x: np.ones([3, 2]),
                                    logits: np.ones([4, 2])})

            sample_dim = tf.placeholder(tf.int32, shape=())
            with pytest.raises(tf.errors.InvalidArgumentError):
                sess.run(bernoulli.logpmf(np.ones([1, 2, 1]), np.ones([1, 2]),
                                          sample_dim=sample_dim),
                         feed_dict={sample_dim: -1})

    def test_logpmf(self):
        with tf.Session() as sess:
            x = [[1, 0], [1, 1]]
            logits = np.array([[-3, -3], [-3, -3]], dtype='float32')
            test_values = sess.run(bernoulli.logpmf(x, logits))
            true_values = stats.bernoulli.logpmf(
                x, 1. / (1. + np.exp(-logits)))
            assert test_values.shape == (2, 2)
            assert np.abs(test_values - true_values).max() < 1e-6

            x = [0, 1]
            logits = np.array([-200, 200], dtype='float32')
            test_values = sess.run(bernoulli.logpmf(x, logits))
            true_values = stats.bernoulli.logpmf(
                x, 1. / (1. + np.exp(-logits)))
            assert test_values.shape == (2,)
            assert np.abs(test_values - true_values).max() < 1e-8

            logits = np.ones([2, 3], dtype='float32')
            x = np.random.randint(2, size=[2, 3, 4]).astype('float32')
            test_value = sess.run(
                bernoulli.logpmf(x, logits, sample_dim=2))
            true_value = stats.bernoulli.logpmf(
                x, 1. / (1. + np.exp(-np.expand_dims(logits, 2))))
            assert test_value.shape == (2, 3, 4)
            assert np.abs(test_value - true_value).max() < 1e-6


class TestDiscrete:
    def test_rvs_static_shape(self):
        logits = tf.zeros([2, 3])
        a = discrete.rvs(logits)
        assert a.get_shape().as_list() == [2, 3]
        with tf.Session() as sess:
            assert sess.run(a).shape == (2, 3)

        b = discrete.rvs(logits, sample_dim=0, n_samples=4)
        assert b.get_shape().as_list() == [4, 2, 3]
        with tf.Session() as sess:
            assert sess.run(b).shape == (4, 2, 3)

        c = discrete.rvs(logits, sample_dim=1, n_samples=4)
        assert c.get_shape().as_list() == [2, 4, 3]
        with tf.Session() as sess:
            assert sess.run(c).shape == (2, 4, 3)

        sample_dim = tf.placeholder(tf.int32, shape=())
        e = discrete.rvs(logits, sample_dim=sample_dim, n_samples=2)
        assert e.get_shape().as_list() == [None, None, None]

        with pytest.raises(ValueError):
            discrete.rvs(logits, sample_dim=-1, n_samples=4)

    def test_rvs(self):
        with tf.Session() as sess:
            logits = tf.placeholder(tf.float32, [None, 3])
            a = sess.run(
                discrete.rvs(logits, sample_dim=0, n_samples=3),
                feed_dict={logits: np.ones([1, 3])})
            assert a.shape == (3, 1, 3)
            assert np.all((np.abs(a - 1) < 1e-8) | (np.abs(a) < 1e-8))
            assert np.max(np.abs(a.sum(axis=-1) - 1)) < 1e-8

            b = sess.run(
                discrete.rvs(logits),
                feed_dict={logits: np.ones([1, 3])}
            )
            assert b.shape == (1, 3)
            assert np.all((np.abs(b - 1) < 1e-8) | (np.abs(b) < 1e-8))
            assert np.max(np.abs(b.sum(axis=-1) - 1)) < 1e-8

            d = sess.run(
                discrete.rvs(logits, sample_dim=1, n_samples=4),
                feed_dict={logits: np.ones([2, 3])})
            assert d.shape == (2, 4, 3)
            assert np.all((np.abs(d - 1) < 1e-8) | (np.abs(d) < 1e-8))
            assert np.max(np.abs(d.sum(axis=-1) - 1)) < 1e-8

            e = sess.run(
                discrete.rvs(logits, sample_dim=2, n_samples=4),
                feed_dict={logits: np.ones([2, 3])})
            assert e.shape == (2, 3, 4)
            assert np.all((np.abs(e - 1) < 1e-8) | (np.abs(e) < 1e-8))
            assert np.max(np.abs(e.sum(axis=-2) - 1)) < 1e-8

            logits = np.array([[2, 5, 3]], dtype='float32')
            p = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
            c = sess.run(discrete.rvs(logits, sample_dim=0, n_samples=1000))
            assert np.max(np.abs(c.mean(axis=0) - p)) < 0.1

            with pytest.raises(tf.errors.InvalidArgumentError):
                sess.run(discrete.rvs(np.ones([1, 3]),
                                      sample_dim=None, n_samples=2))

            sample_dim = tf.placeholder(tf.int32, shape=())
            with pytest.raises(tf.errors.InvalidArgumentError):
                sess.run(discrete.rvs(np.ones([1, 3]),
                                      sample_dim=sample_dim, n_samples=1),
                         feed_dict={sample_dim: -1})

    def test_logpmf_static_shape(self):
        logits = tf.ones([2, 3])
        a = tf.ones([2, 3]) - tf.constant([1, 1, 0], dtype=tf.float32)
        a_logpmf = discrete.logpmf(a, logits)
        assert a_logpmf.get_shape().as_list() == [2]
        with tf.Session() as sess:
            assert sess.run(a_logpmf).shape == (2,)

        b = tf.ones([4, 2, 3]) - tf.constant([1, 1, 0], dtype=tf.float32)
        b_logpmf = discrete.logpmf(b, logits, sample_dim=0)
        assert b_logpmf.get_shape().as_list() == [4, 2]
        with tf.Session() as sess:
            assert sess.run(b_logpmf).shape == (4, 2)

        c = tf.ones([2, 4, 3]) - tf.constant([1, 1, 0], dtype=tf.float32)
        c_logpmf = discrete.logpmf(c, logits, sample_dim=1)
        assert c_logpmf.get_shape().as_list() == [2, 4]
        with tf.Session() as sess:
            assert sess.run(c_logpmf).shape == (2, 4)

        with pytest.raises(ValueError):
            discrete.logpmf(c, logits, sample_dim=-2)

    def test_logpmf_check_shape(self):
        with tf.Session() as sess:
            with pytest.raises(ValueError):
                sess.run(discrete.logpmf(tf.ones([3, 2]),
                                         tf.zeros([1, 3, 2])))

            with pytest.raises(tf.errors.InvalidArgumentError):
                x = tf.placeholder(tf.float32, [None, 2])
                logits = tf.placeholder(tf.float32, [None, 2])
                sess.run(discrete.logpmf(x, logits),
                         feed_dict={x: np.ones([3, 2]),
                                    logits: np.ones([4, 2])})

            sample_dim = tf.placeholder(tf.int32, shape=())
            with pytest.raises(tf.errors.InvalidArgumentError):
                sess.run(discrete.logpmf(np.ones([1, 2, 1]), np.ones([1, 2]),
                                         sample_dim=sample_dim),
                         feed_dict={sample_dim: -1})

    def test_logpmf(self):
        with tf.Session() as sess:
            logits = np.array([[-5, 7., 1.], [-5, 7., 1.]])
            p = np.exp(logits - misc.logsumexp(logits, axis=1, keepdims=True))
            x = np.array([[0, 1, 0], [1, 0, 0]])
            test_values = sess.run(discrete.logpmf(x, logits))
            true_values = np.sum(x * np.log(p), axis=-1)
            assert np.abs(test_values - true_values).max() < 1e-6

            logits = np.array([[0., 2., 3.], [-1., -2., -3.]])
            p = np.exp(logits - misc.logsumexp(logits, axis=1, keepdims=True))
            x = np.array([[[1, 0, 0], [0, 1, 0]],
                          [[0, 0, 1], [0, 0, 1]]])
            test_values = sess.run(discrete.logpmf(x, logits, sample_dim=1))
            true_values = np.sum(x * np.log(np.expand_dims(p, 1)), axis=-1)
            assert test_values.shape == (2, 2)
            assert np.abs(test_values - true_values).max() < 1e-6

            x = np.array([[[1, 0], [0, 0], [0, 1]], [[0, 0], [1, 0], [0, 1]]])
            with pytest.raises(tf.errors.InvalidArgumentError):
                sess.run(discrete.logpmf(x, logits, sample_dim=2))

            x = [0, 1]
            logits = np.array([-200, 200], dtype='float32')
            test_values = sess.run(discrete.logpmf(x, logits))
            true_values = np.array(0, dtype='float32')
            assert test_values.shape == ()
            assert np.abs(test_values - true_values).max() < 1e-8
