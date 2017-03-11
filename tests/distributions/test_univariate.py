#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
from scipy import stats

from tests.context import zhusuan
from zhusuan.distributions.univariate import *


class TestNormal(tf.test.TestCase):
    def test_init_check_shape(self):
        with self.test_session(use_gpu=True):
            with self.assertRaisesRegexp(ValueError,
                                         "should be broadcastable to match"):
                Normal(mean=tf.ones([2, 1]), logstd=tf.zeros([2, 4, 3]))

        Normal(tf.placeholder(tf.float32, [None, 1]),
               tf.placeholder(tf.float32, [None, 1, 3]))

    def test_value_shape(self):
        # static
        norm = Normal(mean=tf.placeholder(tf.float32, None),
                      logstd=tf.placeholder(tf.float32, None))
        self.assertEqual(norm.get_value_shape().as_list(), [])

        # dynamic
        with self.test_session(use_gpu=True):
            self.assertEqual(norm._value_shape().eval().tolist(), [])

    def test_batch_shape(self):
        # static
        def _test_static(mean_shape, logstd_shape, target_shape):
            mean = tf.placeholder(tf.float32, mean_shape)
            logstd = tf.placeholder(tf.float32, logstd_shape)
            norm = Normal(mean, logstd)
            if norm.get_batch_shape():
                self.assertEqual(norm.get_batch_shape().as_list(),
                                 target_shape)
            else:
                self.assertEqual(None, target_shape)

        _test_static([2, 3], [], [2, 3])
        _test_static([2, 3], [3], [2, 3])
        _test_static([2, 1, 4], [2, 3, 4], [2, 3, 4])
        _test_static([2, 3, 5], [3, 1], [2, 3, 5])
        _test_static([1, 2, 3], [1, 3], [1, 2, 3])
        _test_static([None, 3, 5], [3, None], [None, 3, 5])
        _test_static([None, 1, 3], [None, 1], [None, None, 3])
        _test_static([2, None], [], [2, None])
        _test_static(None, [1, 2], None)

        # dynamic
        with self.test_session(use_gpu=True):
            def _test_dynamic(mean_shape, logstd_shape, target_shape):
                mean = tf.placeholder(tf.float32, None)
                logstd = tf.placeholder(tf.float32, None)
                norm = Normal(mean, logstd)
                self.assertEqual(
                    norm.batch_shape.eval(
                        feed_dict={mean: np.ones(mean_shape),
                                   logstd: np.ones(logstd_shape)}).tolist(),
                    target_shape)

            _test_dynamic([2, 3], [], [2, 3])
            _test_dynamic([2, 3], [3], [2, 3])
            _test_dynamic([2, 1, 4], [2, 3, 4], [2, 3, 4])
            _test_dynamic([2, 3, 5], [3, 1], [2, 3, 5])
            with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                         "Incompatible shapes"):
                _test_dynamic([2, 3, 5], [3, 2], None)

    def test_sample_shape(self):
        def _test_static(mean_shape, logstd_shape, n_samples, target_shape):
            mean = tf.placeholder(tf.float32, mean_shape)
            logstd = tf.placeholder(tf.float32, logstd_shape)
            norm = Normal(mean, logstd)
            samples = norm.sample(n_samples)
            if samples.get_shape():
                self.assertEqual(samples.get_shape().as_list(), target_shape)
            else:
                self.assertEqual(None, target_shape)

        _test_static([2, 3], [], 1, [2, 3])
        _test_static([5], [5], 2, [2, 5])
        _test_static([None, 2], [3, None], tf.placeholder(tf.int32, []),
                     None)
        _test_static(None, [1, 2], 1, None)
        _test_static([3, None], [3, 1], 2, [2, 3, None])

        with self.test_session(use_gpu=True):
            def _test_dynamic(mean_shape, logstd_shape, n_samples,
                              target_shape):
                mean = tf.placeholder(tf.float32, None)
                logstd = tf.placeholder(tf.float32, None)
                norm = Normal(mean, logstd)
                samples = norm.sample(n_samples)
                self.assertEqual(
                    tf.shape(samples).eval(
                        feed_dict={mean: np.zeros(mean_shape),
                                   logstd: np.zeros(logstd_shape)}).tolist(),
                    target_shape)

            _test_dynamic([2, 3], [2, 1], 1, [2, 3])
            _test_dynamic([1, 3], [], 2, [2, 1, 3])
            _test_dynamic([2, 1, 5], [3, 1], 3, [3, 2, 3, 5])
            with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                         "Incompatible shapes"):
                _test_dynamic([2, 3, 5], [2, 1], 1, None)

    def test_sample_reparameterized(self):
        mean = tf.ones([2, 3])
        logstd = tf.ones([2, 3])
        norm_rep = Normal(mean, logstd)
        samples = norm_rep.sample(tf.placeholder(tf.int32, shape=[]))
        mean_grads, logstd_grads = tf.gradients(samples, [mean, logstd])
        self.assertTrue(mean_grads is not None)
        self.assertTrue(logstd_grads is not None)

        norm_no_rep = Normal(mean, logstd, is_reparameterized=False)
        samples = norm_no_rep.sample(tf.placeholder(tf.int32, shape=[]))
        mean_grads, logstd_grads = tf.gradients(samples, [mean, logstd])
        self.assertEqual(mean_grads, None)
        self.assertEqual(logstd_grads, None)

    def test_log_prob_shape(self):
        def _test_static(mean_shape, logstd_shape, given_shape, target_shape):
            mean = tf.placeholder(tf.float32, mean_shape)
            logstd = tf.placeholder(tf.float32, logstd_shape)
            given = tf.placeholder(tf.float32, given_shape)
            norm = Normal(mean, logstd)
            log_p = norm.log_prob(given)
            if log_p.get_shape():
                self.assertEqual(log_p.get_shape().as_list(), target_shape)
            else:
                self.assertEqual(None, target_shape)

        _test_static([2, 3], [], [2, 3], [2, 3])
        _test_static([5], [5], [2, 1], [2, 5])
        _test_static([None, 2], [3, None], [None, 1, 1], [None, 3, 2])
        _test_static(None, [1, 2], [2, 2], None)
        _test_static([3, None], [3, 1], [3, 2, 1, 1], [3, 2, 3, None])

        with self.test_session(use_gpu=True):
            def _test_dynamic(mean_shape, logstd_shape, given_shape,
                              target_shape):
                mean = tf.placeholder(tf.float32, None)
                logstd = tf.placeholder(tf.float32, None)
                norm = Normal(mean, logstd)
                given = tf.placeholder(tf.float32, None)
                log_p = norm.log_prob(given)
                self.assertEqual(
                    tf.shape(log_p).eval(
                        feed_dict={mean: np.zeros(mean_shape),
                                   logstd: np.zeros(logstd_shape),
                                   given: np.zeros(given_shape)}).tolist(),
                    target_shape)

            _test_dynamic([2, 3], [2, 1], [1, 3], [2, 3])
            _test_dynamic([1, 3], [], [2, 1, 3], [2, 1, 3])
            _test_dynamic([1, 5], [3, 1], [1, 2, 1, 1], [1, 2, 3, 5])
            with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                         "Incompatible shapes"):
                _test_dynamic([2, 3, 5], [], [1, 2, 1], None)

    def test_value(self):
        with self.test_session(use_gpu=True):
            def _test_value(mean, logstd, given):
                mean = np.array(mean, np.float32)
                logstd = np.array(logstd, np.float32)
                given = np.array(given, np.float32)
                norm = Normal(mean, logstd)
                log_p = norm.log_prob(given)
                target_log_p = stats.norm.logpdf(given, mean, np.exp(logstd))
                self.assertAllClose(log_p.eval(), target_log_p)
                p = norm.prob(given)
                target_p = stats.norm.pdf(given, mean, np.exp(logstd))
                self.assertAllClose(p.eval(), target_p)

            _test_value(0., 0., 0.)
            _test_value(1., [-10., -1., 1., 10.], [0.99, 0.9, 9., 99.])
            _test_value([0., 4.], [[1., 2.], [3., 5.]], [7.])

    def test_check_numerics(self):
        norm = Normal(tf.ones([1, 2]), -1e10, check_numerics=True)
        with self.test_session(use_gpu=True):
            with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                         "precision.*Tensor had Inf"):
                norm.log_prob(0.).eval()


class TestBernoulli(tf.test.TestCase):
    def test_value_shape(self):
        # static
        bernoulli = Bernoulli(tf.placeholder(tf.float32, None))
        self.assertEqual(bernoulli.get_value_shape().as_list(), [])

        # dynamic
        with self.test_session(use_gpu=True):
            self.assertEqual(bernoulli._value_shape().eval().tolist(), [])

    def test_batch_shape(self):
        # static
        def _test_static(logits_shape):
            logits = tf.placeholder(tf.float32, logits_shape)
            bernoulli = Bernoulli(logits)
            if bernoulli.get_batch_shape():
                self.assertEqual(bernoulli.get_batch_shape().as_list(),
                                 logits_shape)
            else:
                self.assertEqual(None, logits_shape)

        _test_static([])
        _test_static([2])
        _test_static([2, 3])
        _test_static([2, 1, 4])
        _test_static([None])
        _test_static([None, 3, 5])
        _test_static([1, None, 3])
        _test_static(None)

        # dynamic
        with self.test_session(use_gpu=True):
            def _test_dynamic(logits_shape):
                logits = tf.placeholder(tf.float32, logits_shape)
                bernoulli = Bernoulli(logits)
                self.assertEqual(
                    bernoulli.batch_shape.eval(
                        feed_dict={logits: np.zeros(logits_shape)}).tolist(),
                    logits_shape)

            _test_dynamic([])
            _test_dynamic([2])
            _test_dynamic([2, 3])
            _test_dynamic([2, 1, 4])

    def test_sample_shape(self):
        def _test_static(logits_shape, n_samples, target_shape):
            logits = tf.placeholder(tf.float32, logits_shape)
            bernoulli = Bernoulli(logits)
            samples = bernoulli.sample(n_samples)
            if samples.get_shape():
                self.assertEqual(samples.get_shape().as_list(), target_shape)
            else:
                self.assertEqual(None, target_shape)

        _test_static([2, 3], 1, [2, 3])
        _test_static([5], 2, [2, 5])
        _test_static([None, 2], tf.placeholder(tf.int32, []), None)
        _test_static(None, 1, None)
        _test_static([3, None], 2, [2, 3, None])

        with self.test_session(use_gpu=True):
            def _test_dynamic(logits_shape, n_samples, target_shape):
                logits = tf.placeholder(tf.float32, None)
                bernoulli = Bernoulli(logits)
                samples = bernoulli.sample(n_samples)
                self.assertEqual(
                    tf.shape(samples).eval(
                        feed_dict={logits: np.zeros(logits_shape)}).tolist(),
                    target_shape)

            _test_dynamic([2, 3], 1, [2, 3])
            _test_dynamic([1, 3], 2, [2, 1, 3])
            _test_dynamic([2, 1, 5], 3, [3, 2, 1, 5])

    def test_log_prob_shape(self):
        def _test_static(logits_shape, given_shape, target_shape):
            logits = tf.placeholder(tf.float32, logits_shape)
            given = tf.placeholder(tf.int32, given_shape)
            bernoulli = Bernoulli(logits)
            log_p = bernoulli.log_prob(given)
            if log_p.get_shape():
                self.assertEqual(log_p.get_shape().as_list(), target_shape)
            else:
                self.assertEqual(None, target_shape)

        _test_static([2, 3], [2, 3], [2, 3])
        _test_static([5], [2, 1], [2, 5])
        _test_static([None, 2], [None, 1, 1], [None, None, 2])
        _test_static(None, [2, 2], None)
        _test_static([3, None], [3, 2, 1, 1], [3, 2, 3, None])

        with self.test_session(use_gpu=True):
            def _test_dynamic(logits_shape, given_shape, target_shape):
                logits = tf.placeholder(tf.float32, None)
                bernoulli = Bernoulli(logits)
                given = tf.placeholder(tf.int32, None)
                log_p = bernoulli.log_prob(given)
                self.assertEqual(
                    tf.shape(log_p).eval(
                        feed_dict={logits: np.zeros(logits_shape),
                                   given: np.zeros(given_shape,
                                                   np.int32)}).tolist(),
                    target_shape)

            _test_dynamic([2, 3], [1, 3], [2, 3])
            _test_dynamic([1, 3], [2, 2, 3], [2, 2, 3])
            _test_dynamic([1, 5], [1, 2, 3, 1], [1, 2, 3, 5])
            with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                         "Incompatible shapes"):
                _test_dynamic([2, 3, 5], [1, 2, 1], None)

    def test_value(self):
        with self.test_session(use_gpu=True):
            def _test_value(logits, given):
                logits = np.array(logits, np.float32)
                given = np.array(given, np.float32)
                bernoulli = Bernoulli(logits)
                log_p = bernoulli.log_prob(given)
                target_log_p = stats.bernoulli.logpmf(
                    given, 1. / (1 + np.exp(-logits)))
                self.assertAllClose(log_p.eval(), target_log_p)
                p = bernoulli.prob(given)
                target_p = stats.bernoulli.pmf(
                    given, 1. / (1 + np.exp(-logits)))
                self.assertAllClose(p.eval(), target_p)

            _test_value(0., [0, 1])
            _test_value([-100., -10., 10., 100.], [1, 1, 0, 0])
            _test_value([0., 4.], [[0, 1], [0, 5]])


class TestCategorical(tf.test.TestCase):
    pass


class TestUniform(tf.test.TestCase):
    pass


class TestGamma(tf.test.TestCase):
    pass


class TestBeta(tf.test.TestCase):
    pass
