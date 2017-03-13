#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
from scipy import stats, misc

from tests.context import zhusuan
from zhusuan.distributions.multivariate import *


class TestMultinomial(tf.test.TestCase):
    def test_init_check_shape(self):
        pass


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
        cat2 = OnehotCategorical(tf.placeholder(tf.float32, None))
        with self.test_session(use_gpu=True):
            self.assertEqual(cat2._value_shape().eval(
                feed_dict={logits: np.ones([2, 1 ,3])}).tolist(), [3])

    def test_batch_shape(self):
        # static
        def _test_static(logits_shape):
            logits = tf.placeholder(tf.float32, logits_shape)
            cat = OnehotCategorical(logits)
            if cat.get_batch_shape():
                self.assertEqual(cat.get_batch_shape().as_list(), logits_shape)
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
                logits = tf.placeholder(tf.float32, logits_shape)
                cat = OnehotCategorical(logits)
                self.assertEqual(
                    cat.batch_shape.eval(
                        feed_dict={logits: np.zeros(logits_shape)}).tolist(),
                    logits_shape)

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

        _test_static([2], 1, [2])
        _test_static([2, 3], 1, [2, 3])
        _test_static([5], 2, [2, 5])
        _test_static([1, 2, 4], 3, [3, 1, 2, 4])
        _test_static([None, 2], tf.placeholder(tf.int32, []), None)
        _test_static(None, 1, None)
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

            _test_dynamic([2], 1, [2])
            _test_dynamic([2, 3], 1, [2, 3])
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

        _test_static([2, 3], [2], [2, 3])
        _test_static([5], [], [5])
        _test_static([1, 2, 4], [1], [1, 2, 4])
        _test_static([3, 1, 5], [1, 4], [3, 4, 5])
        _test_static([None, 2, 4], [3, None, 1], [3, 2, 4])
        _test_static([None, 2], [None, 1, 1, 2], [None, 1, None, 2])
        _test_static(None, [2, 2], None)
        _test_static([3, None], [3, 2, 1, 1], [3, 2, 1, 3])
        with self.assertRaisesRegexp(ValueError, "broadcast to match"):
            _test_static([2, 3, 5], [1, 2], None)

        with self.test_session(use_gpu=True):
            def _test_dynamic(logits_shape, given_shape, target_shape):
                logits = tf.placeholder(tf.float32, None)
                cat = OnehotCategorical(logits)
                given = tf.placeholder(tf.int32, None)
                log_p = cat.log_prob(given)
                self.assertEqual(
                    tf.shape(log_p).eval(
                        feed_dict={logits: np.zeros(logits_shape),
                                   given: np.zeros(given_shape,
                                                   np.int32)}).tolist(),
                    target_shape)

            _test_dynamic([2, 3, 3], [1, 3], [2, 3])
            _test_dynamic([1, 3, 4], [2, 2, 3], [2, 2, 3])
            _test_dynamic([1, 5, 1], [1, 2, 3, 1], [1, 2, 3, 5])
            with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                         "Incompatible shapes"):
                _test_dynamic([2, 3, 5], [1, 2], None)

    def test_value(self):
        with self.test_session(use_gpu=True):
            def _test_value(logits, given):
                logits = np.array(logits, np.float32)
                normalized_logits = logits - misc.logsumexp(
                    logits, axis=-1, keepdims=True)
                given = np.array(given, np.int32)
                cat = OnehotCategorical(logits)
                log_p = cat.log_prob(given)

                def _one_hot(x, depth):
                    n_elements = x.size
                    ret = np.zeros((n_elements, depth))
                    ret[np.arange(n_elements), x.flat] = 1
                    return ret.reshape(list(x.shape) + [depth])

                target_log_p = np.sum(_one_hot(
                    given, logits.shape[-1]) * normalized_logits, -1)
                self.assertAllClose(log_p.eval(), target_log_p)
                p = cat.prob(given)
                target_p = np.sum(_one_hot(
                    given, logits.shape[-1]) * np.exp(normalized_logits), -1)
                self.assertAllClose(p.eval(), target_p)

            _test_value([0.], [0, 0, 0])
            _test_value([-50., -10., -50.], [0, 1, 2, 1])
            _test_value([0., 4.], [[0, 1], [0, 1]])
            _test_value([[2., 3., 1.], [5., 7., 4.]],
                        np.ones([3, 1, 1], dtype=np.int32))
