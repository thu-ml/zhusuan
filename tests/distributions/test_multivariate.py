#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
from scipy import stats, misc, special

from tests.distributions.utils import *
from zhusuan.distributions.multivariate import *


class TestMultinomial(tf.test.TestCase):
    def test_init_check_shape(self):
        with self.test_session(use_gpu=True):
            with self.assertRaisesRegexp(ValueError, "should have rank"):
                Multinomial(tf.zeros([]), 10)

    def test_init_n(self):
        dist = Multinomial(tf.ones([2]), 10)
        self.assertTrue(isinstance(dist.n_categories, int))
        self.assertEqual(dist.n_categories, 2)
        self.assertTrue(isinstance(dist.n_experiments, int))
        self.assertEqual(dist.n_experiments, 10)
        with self.assertRaisesRegexp(ValueError, "must be positive"):
            _ = Multinomial(tf.ones([2]), 0)

        with self.test_session(use_gpu=True) as sess:
            logits = tf.placeholder(tf.float32, None)
            n_experiments = tf.placeholder(tf.int32, None)
            dist2 = Multinomial(logits, n_experiments)
            self.assertEqual(
                sess.run([dist2.n_categories, dist2.n_experiments],
                         feed_dict={logits: np.ones([2]), n_experiments: 10}),
                [2, 10])
            with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                         "should have rank"):
                dist2.n_categories.eval(feed_dict={logits: 1.,
                                                   n_experiments: 10})
            with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                         "should be a scalar"):
                dist2.n_experiments.eval(feed_dict={logits: [1.],
                                                    n_experiments: [10]})
            with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                         "must be positive"):
                dist2.n_experiments.eval(feed_dict={logits: [1.],
                                                    n_experiments: 0})

    def test_value_shape(self):
        # static
        dist = Multinomial(tf.placeholder(tf.float32, [None, 2]), 10)
        self.assertEqual(dist.get_value_shape().as_list(), [2])

        # dynamic
        logits = tf.placeholder(tf.float32, None)
        dist2 = Multinomial(logits, 10)
        self.assertTrue(dist2._value_shape().dtype is tf.int32)
        with self.test_session(use_gpu=True):
            self.assertEqual(dist2._value_shape().eval(
                feed_dict={logits: np.ones([2])}).tolist(), [2])

        self.assertEqual(dist._value_shape().dtype, tf.int32)

    def test_batch_shape(self):
        # static
        def _test_static(logits_shape):
            logits = tf.placeholder(tf.float32, logits_shape)
            dist = Multinomial(logits, 10)
            if dist.get_batch_shape():
                self.assertEqual(dist.get_batch_shape().as_list(),
                                 logits_shape[:-1])
            else:
                self.assertEqual(None, logits_shape)

        _test_static([2])
        _test_static([2, 3])
        _test_static([2, 1, 4])
        _test_static([None])
        _test_static([None, 3, 5])
        _test_static([1, None, 3])
        _test_static([None, 1, 10])
        _test_static(None)

        # dynamic
        with self.test_session(use_gpu=True):
            def _test_dynamic(logits_shape):
                logits = tf.placeholder(tf.float32, None)
                dist = Multinomial(logits, 10)
                self.assertTrue(dist.batch_shape.dtype is tf.int32)
                self.assertEqual(
                    dist.batch_shape.eval(
                        feed_dict={logits: np.zeros(logits_shape)}).tolist(),
                    logits_shape[:-1])

            _test_dynamic([2])
            _test_dynamic([2, 3])
            _test_dynamic([2, 1, 4])

    def test_sample_shape(self):
        def _test_static(logits_shape, n_samples, target_shape):
            logits = tf.placeholder(tf.float32, logits_shape)
            dist = Multinomial(logits, 10)
            samples = dist.sample(n_samples)
            if samples.get_shape():
                self.assertEqual(samples.get_shape().as_list(), target_shape)
            else:
                self.assertEqual(None, target_shape)

        _test_static([2], None, [2])
        _test_static([2], 1, [1, 2])
        _test_static([2, 3], None, [2, 3])
        _test_static([2, 3], 1, [1, 2, 3])
        _test_static([5], 2, [2, 5])
        _test_static([1, 2, 4], 3, [3, 1, 2, 4])
        _test_static([None, 2], tf.placeholder(tf.int32, []), [None, None, 2])
        _test_static(None, None, None)
        _test_static(None, 1, None)
        _test_static([None, 1, 10], None, [None, 1, 10])
        _test_static([3, None], 2, [2, 3, None])

        with self.test_session(use_gpu=True):
            def _test_dynamic(logits_shape, n_samples, target_shape):
                logits = tf.placeholder(tf.float32, None)
                dist = Multinomial(logits, 10)
                samples = dist.sample(n_samples)
                self.assertEqual(
                    tf.shape(samples).eval(
                        feed_dict={logits: np.zeros(logits_shape)}).tolist(),
                    target_shape)

            _test_dynamic([2], 1, [1, 2])
            _test_dynamic([2, 3], 1, [1, 2, 3])
            _test_dynamic([1, 3], 2, [2, 1, 3])
            _test_dynamic([2, 1, 5], 3, [3, 2, 1, 5])

    def test_log_prob_shape(self):
        def _test_static(logits_shape, given_shape, target_shape):
            logits = tf.placeholder(tf.float32, logits_shape)
            given = tf.placeholder(tf.int32, given_shape)
            dist = Multinomial(logits, 1)
            log_p = dist.log_prob(given)
            if log_p.get_shape():
                self.assertEqual(log_p.get_shape().as_list(), target_shape)
            else:
                self.assertEqual(None, target_shape)

        _test_static([2, 3], [2, 3], [2])
        _test_static([2, 5], [5], [2])
        _test_static([1, 2, 4], [4], [1, 2])
        _test_static([3, 1, 5], [1, 4, 5], [3, 4])
        _test_static([1, 4], [2, 5, 4], [2, 5])
        _test_static([None, 2, 4], [3, None, 4], [3, 2])
        _test_static([None, 2], [None, 1, 1, 2], [None, 1, None])
        _test_static(None, [2, 2], None)
        _test_static([3, None], [3, 2, 1, 1], [3, 2, 3])
        with self.assertRaisesRegexp(ValueError, "broadcast to match"):
            _test_static([2, 3, 5], [1, 2, 5], None)

        with self.test_session(use_gpu=True):
            def _test_dynamic(logits_shape, given_shape, target_shape):
                logits = tf.placeholder(tf.float32, None)
                n_experiments = tf.placeholder(tf.int32, [])
                dist = Multinomial(logits, n_experiments)
                given = tf.placeholder(tf.int32, None)
                log_p = dist.log_prob(given)

                def _make_samples(shape):
                    samples = np.zeros(shape)
                    samples = samples.reshape((-1, shape[-1]))
                    samples[:, 0] = 1
                    return samples.reshape(shape)

                logits_ = _make_samples(logits_shape)
                given_ = _make_samples(given_shape)
                self.assertEqual(
                    tf.shape(log_p).eval(
                        feed_dict={logits: logits_,
                                   n_experiments: 1,
                                   given: given_}).tolist(),
                    target_shape)

            _test_dynamic([2, 3, 3], [1, 3], [2, 3])
            _test_dynamic([1, 3], [2, 2, 3], [2, 2])
            _test_dynamic([1, 5, 1], [1, 2, 1, 1], [1, 2, 5])
            with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                         "Incompatible shapes"):
                _test_dynamic([2, 3, 5], [1, 2, 5], None)

    def test_value(self):
        with self.test_session(use_gpu=True):
            def _test_value(logits, n_experiments, given):
                logits = np.array(logits, np.float32)
                normalized_logits = logits - misc.logsumexp(
                    logits, axis=-1, keepdims=True)
                given = np.array(given)
                dist = Multinomial(logits, n_experiments)
                log_p = dist.log_prob(given)
                target_log_p = np.log(misc.factorial(n_experiments)) - \
                    np.sum(np.log(misc.factorial(given)), -1) + \
                    np.sum(given * normalized_logits, -1)
                self.assertAllClose(log_p.eval(), target_log_p)
                p = dist.prob(given)
                target_p = np.exp(target_log_p)
                self.assertAllClose(p.eval(), target_p)

            _test_value([-50., -20., 0.], 4, [1, 0, 3])
            _test_value([1., 10., 1000.], 1, [1, 0, 0])
            _test_value([[2., 3., 1.], [5., 7., 4.]], 3,
                        np.ones([3, 1, 3], dtype=np.int32))
            _test_value([-10., 10., 20., 50.], 100, [[0, 1, 99, 100],
                                                     [100, 99, 1, 0]])

    def test_dtype(self):
        def _distribution(param, dtype=None):
            return Multinomial(param, 10, dtype)
        test_dtype_1parameter_discrete(self, _distribution)

        with self.assertRaisesRegexp(TypeError, "n_experiments must be"):
            Multinomial([1., 1.], tf.placeholder(tf.float32, []))


class TestOnehotCategorical(tf.test.TestCase):
    def test_init_check_shape(self):
        with self.test_session(use_gpu=True):
            with self.assertRaisesRegexp(ValueError, "should have rank"):
                OnehotCategorical(logits=tf.zeros([]))

    def test_init_n_categories(self):
        cat = OnehotCategorical(tf.ones([10]))
        self.assertTrue(isinstance(cat.n_categories, int))
        self.assertEqual(cat.n_categories, 10)

        with self.test_session(use_gpu=True):
            logits = tf.placeholder(tf.float32, None)
            cat2 = OnehotCategorical(logits)
            self.assertEqual(
                cat2.n_categories.eval(feed_dict={logits: np.ones([10])}), 10)
            with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                         "should have rank"):
                cat2.n_categories.eval(feed_dict={logits: 1.})

    def test_value_shape(self):
        # static
        cat = OnehotCategorical(tf.placeholder(tf.float32, [None, 10]))
        self.assertEqual(cat.get_value_shape().as_list(), [10])

        # dynamic
        logits = tf.placeholder(tf.float32, None)
        cat2 = OnehotCategorical(logits)
        self.assertTrue(cat2._value_shape().dtype is tf.int32)
        with self.test_session(use_gpu=True):
            self.assertEqual(cat2._value_shape().eval(
                feed_dict={logits: np.ones([2, 1, 3])}).tolist(), [3])

        self.assertEqual(cat._value_shape().dtype, tf.int32)

    def test_batch_shape(self):
        # static
        def _test_static(logits_shape):
            logits = tf.placeholder(tf.float32, logits_shape)
            cat = OnehotCategorical(logits)
            if cat.get_batch_shape():
                self.assertEqual(cat.get_batch_shape().as_list(),
                                 logits_shape[:-1])
            else:
                self.assertEqual(None, logits_shape)

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
                logits = tf.placeholder(tf.float32, None)
                cat = OnehotCategorical(logits)
                self.assertTrue(cat.batch_shape.dtype is tf.int32)
                self.assertEqual(
                    cat.batch_shape.eval(
                        feed_dict={logits: np.zeros(logits_shape)}).tolist(),
                    logits_shape[:-1])

            _test_dynamic([2])
            _test_dynamic([2, 3])
            _test_dynamic([2, 1, 4])

    def test_sample_shape(self):
        def _test_static(logits_shape, n_samples, target_shape):
            logits = tf.placeholder(tf.float32, logits_shape)
            cat = OnehotCategorical(logits)
            samples = cat.sample(n_samples)
            if samples.get_shape():
                self.assertEqual(samples.get_shape().as_list(), target_shape)
            else:
                self.assertEqual(None, target_shape)

        _test_static([2], None, [2])
        _test_static([2], 1, [1, 2])
        _test_static([2, 3], None, [2, 3])
        _test_static([2, 3], 1, [1, 2, 3])
        _test_static([5], 2, [2, 5])
        _test_static([1, 2, 4], 3, [3, 1, 2, 4])
        _test_static([None, 2], tf.placeholder(tf.int32, []), [None, None, 2])
        _test_static(None, None, None)
        _test_static(None, 1, None)
        _test_static([None, 1, 10], None, [None, 1, 10])
        _test_static([3, None], 2, [2, 3, None])

        with self.test_session(use_gpu=True):
            def _test_dynamic(logits_shape, n_samples, target_shape):
                logits = tf.placeholder(tf.float32, None)
                cat = OnehotCategorical(logits)
                samples = cat.sample(n_samples)
                self.assertEqual(
                    tf.shape(samples).eval(
                        feed_dict={logits: np.zeros(logits_shape)}).tolist(),
                    target_shape)

            _test_dynamic([2], 1, [1, 2])
            _test_dynamic([2, 3], 1, [1, 2, 3])
            _test_dynamic([1, 3], 2, [2, 1, 3])
            _test_dynamic([2, 1, 5], 3, [3, 2, 1, 5])

    def test_log_prob_shape(self):
        def _test_static(logits_shape, given_shape, target_shape):
            logits = tf.placeholder(tf.float32, logits_shape)
            given = tf.placeholder(tf.int32, given_shape)
            cat = OnehotCategorical(logits)
            log_p = cat.log_prob(given)
            if log_p.get_shape():
                self.assertEqual(log_p.get_shape().as_list(), target_shape)
            else:
                self.assertEqual(None, target_shape)

        _test_static([2, 3], [2, 3], [2])
        _test_static([2, 5], [5], [2])
        _test_static([1, 2, 4], [4], [1, 2])
        _test_static([3, 1, 5], [1, 4, 5], [3, 4])
        _test_static([1, 4], [2, 5, 4], [2, 5])
        _test_static([None, 2, 4], [3, None, 4], [3, 2])
        _test_static([None, 2], [None, 1, 1, 2], [None, 1, None])
        _test_static(None, [2, 2], None)
        _test_static([3, None], [3, 2, 1, 1], [3, 2, 3])
        with self.assertRaisesRegexp(ValueError, "broadcast to match"):
            _test_static([2, 3, 5], [1, 2, 5], None)

        with self.test_session(use_gpu=True):
            def _test_dynamic(logits_shape, given_shape, target_shape):
                logits = tf.placeholder(tf.float32, None)
                dist = OnehotCategorical(logits)
                given = tf.placeholder(tf.int32, None)
                log_p = dist.log_prob(given)

                def _make_valid_samples(shape):
                    samples = np.zeros(shape)
                    samples = samples.reshape((-1, shape[-1]))
                    samples[:, 0] = 1
                    return samples.reshape(shape)

                logits_ = _make_valid_samples(logits_shape)
                given_ = _make_valid_samples(given_shape)
                self.assertEqual(
                    tf.shape(log_p).eval(
                        feed_dict={logits: logits_,
                                   given: given_}).tolist(),
                    target_shape)

            _test_dynamic([2, 3, 3], [1, 3], [2, 3])
            _test_dynamic([1, 3], [2, 2, 3], [2, 2])
            _test_dynamic([1, 5, 1], [1, 2, 1, 1], [1, 2, 5])
            with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                         "Incompatible shapes"):
                _test_dynamic([2, 3, 5], [1, 2, 5], None)

    def test_value(self):
        with self.test_session(use_gpu=True):
            def _test_value(logits, given):
                logits = np.array(logits, np.float32)
                normalized_logits = logits - misc.logsumexp(
                    logits, axis=-1, keepdims=True)
                given = np.array(given, np.int32)
                cat = OnehotCategorical(logits)
                log_p = cat.log_prob(tf.one_hot(given, logits.shape[-1],
                                                dtype=tf.int32))

                def _one_hot(x, depth):
                    n_elements = x.size
                    ret = np.zeros((n_elements, depth))
                    ret[np.arange(n_elements), x.flat] = 1
                    return ret.reshape(list(x.shape) + [depth])

                target_log_p = np.sum(_one_hot(
                    given, logits.shape[-1]) * normalized_logits, -1)
                self.assertAllClose(log_p.eval(), target_log_p)
                p = cat.prob(tf.one_hot(given, logits.shape[-1],
                                        dtype=tf.int32))
                target_p = np.sum(_one_hot(
                    given, logits.shape[-1]) * np.exp(normalized_logits), -1)
                self.assertAllClose(p.eval(), target_p)

            _test_value([0.], [0, 0, 0])
            _test_value([-50., -10., -50.], [0, 1, 2, 1])
            _test_value([0., 4.], [[0, 1], [0, 1]])
            _test_value([[2., 3., 1.], [5., 7., 4.]],
                        np.ones([3, 1, 1], dtype=np.int32))

    def test_dtype(self):
        test_dtype_1parameter_discrete(self, OnehotCategorical)


class TestDirichlet(tf.test.TestCase):
    def test_init_check_shape(self):
        with self.test_session(use_gpu=True):
            with self.assertRaisesRegexp(ValueError, "should have rank"):
                Dirichlet(alpha=tf.zeros([]))

    def test_init_n_categories(self):
        dist = Dirichlet(tf.ones([10]))
        self.assertTrue(isinstance(dist.n_categories, int))
        self.assertEqual(dist.n_categories, 10)
        with self.assertRaisesRegexp(ValueError,
                                     "n_categories.*should be at least 2"):
            Dirichlet(tf.ones([3, 1]))
        dist2 = Dirichlet(tf.placeholder(tf.float32, [3, None]))
        self.assertTrue(dist2.n_categories is not None)

        with self.test_session(use_gpu=True):
            alpha = tf.placeholder(tf.float32, None)
            dist3 = Dirichlet(alpha)
            self.assertEqual(
                dist3.n_categories.eval(feed_dict={alpha: np.ones([10])}), 10)
            with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                         "should have rank"):
                dist3.n_categories.eval(feed_dict={alpha: 1.})

    def test_value_shape(self):
        # static
        dist = Dirichlet(tf.placeholder(tf.float32, [None, 10]))
        self.assertEqual(dist.get_value_shape().as_list(), [10])

        # dynamic
        alpha = tf.placeholder(tf.float32, None)
        dist2 = Dirichlet(alpha)
        self.assertEqual(dist2.get_value_shape().as_list(), [None])
        self.assertTrue(dist2._value_shape().dtype is tf.int32)
        with self.test_session(use_gpu=True):
            self.assertEqual(dist2._value_shape().eval(
                feed_dict={alpha: np.ones([2, 1, 3])}).tolist(), [3])
        self.assertEqual(dist._value_shape().dtype, tf.int32)

    def test_batch_shape(self):
        # static
        def _test_static(alpha_shape):
            alpha = tf.placeholder(tf.float32, alpha_shape)
            dist = Dirichlet(alpha)
            if dist.get_batch_shape():
                self.assertEqual(dist.get_batch_shape().as_list(),
                                 alpha_shape[:-1])
            else:
                self.assertEqual(None, alpha_shape)

        _test_static([2])
        _test_static([2, 3])
        _test_static([2, 1, 4])
        _test_static([None])
        _test_static([None, 3, 5])
        _test_static([1, None, 3])
        _test_static(None)

        # dynamic
        with self.test_session(use_gpu=True):
            def _test_dynamic(alpha_shape):
                alpha = tf.placeholder(tf.float32, None)
                dist = Dirichlet(alpha)
                self.assertTrue(dist.batch_shape.dtype is tf.int32)
                self.assertEqual(
                    dist.batch_shape.eval(
                        feed_dict={alpha: np.zeros(alpha_shape)}).tolist(),
                    alpha_shape[:-1])

            _test_dynamic([2])
            _test_dynamic([2, 3])
            _test_dynamic([2, 1, 4])

    def test_sample_shape(self):
        def _test_static(alpha_shape, n_samples, target_shape):
            alpha = tf.placeholder(tf.float32, alpha_shape)
            dist = Dirichlet(alpha)
            samples = dist.sample(n_samples)
            if samples.get_shape():
                self.assertEqual(samples.get_shape().as_list(), target_shape)
            else:
                self.assertEqual(None, target_shape)

        _test_static([2], None, [2])
        _test_static([2], 1, [1, 2])
        _test_static([2, 3], None, [2, 3])
        _test_static([2, 3], 1, [1, 2, 3])
        _test_static([5], 2, [2, 5])
        _test_static([1, 2, 4], 3, [3, 1, 2, 4])
        _test_static([None, 2], tf.placeholder(tf.int32, []), [None, None, 2])
        _test_static(None, None, None)
        _test_static(None, 1, None)
        _test_static([None, 1, 10], None, [None, 1, 10])
        _test_static([3, None], 2, [2, 3, None])

        with self.test_session(use_gpu=True):
            def _test_dynamic(alpha_shape, n_samples, target_shape):
                alpha = tf.placeholder(tf.float32, None)
                dist = Dirichlet(alpha)
                samples = dist.sample(n_samples)
                self.assertEqual(
                    tf.shape(samples).eval(
                        feed_dict={alpha: np.zeros(alpha_shape)}).tolist(),
                    target_shape)

            _test_dynamic([2], 1, [1, 2])
            _test_dynamic([2, 3], 1, [1, 2, 3])
            _test_dynamic([1, 3], 2, [2, 1, 3])
            _test_dynamic([2, 1, 5], 3, [3, 2, 1, 5])

    def test_log_prob_shape(self):
        def _test_static(alpha_shape, given_shape, target_shape):
            alpha = tf.placeholder(tf.float32, alpha_shape)
            given = tf.placeholder(tf.float32, given_shape)
            dist = Dirichlet(alpha)
            log_p = dist.log_prob(given)
            if log_p.get_shape():
                self.assertEqual(log_p.get_shape().as_list(), target_shape)
            else:
                self.assertEqual(None, target_shape)

        _test_static([2, 3], [2, 3], [2])
        _test_static([2, 5], [5], [2])
        _test_static([1, 2, 4], [4], [1, 2])
        _test_static([3, 1, 5], [1, 4, 5], [3, 4])
        _test_static([1, 4], [2, 5, 4], [2, 5])
        _test_static([None, 2, 4], [3, None, 4], [3, 2])
        _test_static([None, 2], [None, 1, 1, 2], [None, 1, None])
        _test_static(None, [2, 2], None)
        # TODO: This failed with a bug in Tensorflow, waiting fix.
        # _test_static([3, None], [3, 2, 1, None], [3, 2, 3])
        with self.assertRaisesRegexp(ValueError, "broadcast to match"):
            _test_static([2, 3, 5], [1, 2, 5], None)

        with self.test_session(use_gpu=True):
            def _test_dynamic(alpha_shape, given_shape, target_shape):
                alpha = tf.placeholder(tf.float32, None)
                dist = Dirichlet(alpha)
                given = tf.placeholder(tf.float32, None)
                log_p = dist.log_prob(given)

                def _make_valid_samples(shape):
                    samples = np.ones(shape, dtype=np.float32)
                    return samples / samples.sum(axis=-1, keepdims=True)

                alpha_ = np.ones(alpha_shape)
                given_ = _make_valid_samples(given_shape)
                self.assertEqual(
                    tf.shape(log_p).eval(
                        feed_dict={alpha: alpha_,
                                   given: given_}).tolist(),
                    target_shape)

            _test_dynamic([2, 3, 3], [1, 3], [2, 3])
            _test_dynamic([1, 3], [2, 2, 3], [2, 2])
            _test_dynamic([1, 5, 2], [1, 2, 1, 2], [1, 2, 5])
            with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                         "Incompatible shapes"):
                _test_dynamic([2, 3, 5], [1, 2, 5], None)

    def test_value(self):
        def dirichlet_logpdf(x, alpha):
            # scipy's implementation of dirichlet logpdf doesn't support
            # batch of x, we use this modified version.
            def _lnB(alpha):
                return np.sum(special.gammaln(alpha)) - \
                    special.gammaln(np.sum(alpha))

            lnB = _lnB(alpha)
            return - lnB + np.sum(np.log(x) * (alpha - 1), -1)

        def dirichlet_pdf(x, alpha):
            return np.exp(dirichlet_logpdf(x, alpha))

        with self.test_session(use_gpu=True):
            def _test_value_alpha_rank1(alpha, given):
                alpha = np.array(alpha, np.float32)
                given = np.array(given, np.float32)
                dist = Dirichlet(alpha)
                log_p = dist.log_prob(given)
                target_log_p = dirichlet_logpdf(given, alpha)
                self.assertAllClose(log_p.eval(), target_log_p)
                p = dist.prob(given)
                target_p = dirichlet_pdf(given, alpha)
                self.assertAllClose(p.eval(), target_p)

            _test_value_alpha_rank1([1., 1., 1.],
                                    [[0.2, 0.5, 0.3], [0.3, 0.4, 0.3]])
            _test_value_alpha_rank1([2., 3., 4.], [0.3, 0.7, 0.])
            # TODO: fix for case when alpha=1, given=0

            def _test_value_alpha_rank2_given_rank2(alpha, given):
                alpha = np.array(alpha, np.float32)
                given = np.array(given, np.float32)
                alpha_b = alpha * np.ones_like(given)
                given_b = given * np.ones_like(alpha)
                dist = Dirichlet(alpha)
                log_p = dist.log_prob(given)
                target_log_p = np.array(
                    [dirichlet_logpdf(given_b[i], alpha_b[i])
                     for i in range(alpha_b.shape[0])])
                self.assertAllClose(log_p.eval(), target_log_p)
                p = dist.prob(given)
                target_p = np.array(
                    [dirichlet_pdf(given_b[i], alpha_b[i])
                     for i in range(alpha_b.shape[0])])
                self.assertAllClose(p.eval(), target_p)

            _test_value_alpha_rank2_given_rank2([[1., 2.], [3., 4.]],
                                                [0.5, 0.5])
            _test_value_alpha_rank2_given_rank2([[5., 6.], [7., 8.]],
                                                [[0.1, 0.9]])
            _test_value_alpha_rank2_given_rank2([[100., 1.], [0.01, 10.]],
                                                [[0., 1.], [1., 0.]])

    def test_check_numerics(self):
        alpha = tf.placeholder(tf.float32, None)
        given = tf.placeholder(tf.float32, None)
        dist = Dirichlet(alpha, check_numerics=True)
        log_p = dist.log_prob(given)
        with self.test_session(use_gpu=True):
            with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                         "log\(given\).*Tensor had Inf"):
                log_p.eval(feed_dict={alpha: np.ones([2]), given: [0., 1.]})
            with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                         "lbeta\(alpha\).*Tensor had NaN"):
                log_p.eval(feed_dict={alpha: [-1., 1.], given: [0.5, 0.5]})

    def test_dtype(self):
        test_dtype_1parameter_continuous(self, Dirichlet)
