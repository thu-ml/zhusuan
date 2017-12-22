#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np

from tests.distributions import utils
from zhusuan.distributions.special import *


class TestEmpirical(tf.test.TestCase):
    def test_init(self):
        Empirical(batch_shape=[None, 1], dtype=tf.float32)
        Empirical(batch_shape=[None, 10], value_shape=5, dtype=tf.float32)
        Empirical(batch_shape=[None, 1], dtype=tf.int32)
        Empirical(batch_shape=[None, 10], value_shape=5, dtype=tf.int32)

    def test_shape(self):
        def _test_shape(batch_shape, value_shape):
            empirical = Empirical(batch_shape, tf.float32, value_shape)
            if value_shape is None:
                value_shape = []
            elif not isinstance(value_shape, (list, tuple)):
                value_shape = [value_shape]

            # static
            self.assertEqual(empirical.get_batch_shape().as_list(), batch_shape)
            self.assertEqual(empirical.get_value_shape().as_list(), value_shape)

            # dynamic
            if None not in batch_shape:
                self.assertTrue(empirical._batch_shape().dtype is tf.int32)
                with self.test_session(use_gpu=True):
                    self.assertEqual(empirical._batch_shape().eval().tolist(), batch_shape)

            self.assertTrue(empirical._value_shape().dtype is tf.int32)
            with self.test_session(use_gpu=True):
                self.assertEqual(empirical._value_shape().eval().tolist(), value_shape)

        # No value shape
        _test_shape([None, 1], None)
        _test_shape([12, 1], None)
        _test_shape([None, 10, 15], None)
        _test_shape([12, 10, 15], None)
        # With value shape
        _test_shape([None, 1], 5)
        _test_shape([12, 1], 5)
        _test_shape([None, 10, 15], [5, 3])
        _test_shape([12, 10, 15], [5, 3])

    def test_sample(self):
        with self.assertRaisesRegexp(
                ValueError, "You can not sample from an Empirical distribution."):
            Empirical(batch_shape=[None, 1], dtype=tf.float32).sample()

    def test_log_prob(self):
        with self.assertRaisesRegexp(
                ValueError, "An empirical distribution has no log-probability measure."):
            Empirical(batch_shape=[None, 1], dtype=tf.float32).log_prob(np.zeros((5, 1)))

    def test_prob(self):
        with self.assertRaisesRegexp(
                ValueError, "An empirical distribution has no probability measure."):
            Empirical(batch_shape=[None, 1], dtype=tf.float32).prob(np.zeros((5, 1)))

    def test_dtype(self):
        def _test_dtype(batch_shape, dtype):
            self.assertEqual(Empirical(batch_shape=batch_shape, dtype=dtype).dtype,
                             dtype)

        _test_dtype([None, 1], tf.float16)
        _test_dtype([None, 1], tf.float32)
        _test_dtype([None, 1], tf.float64)
        _test_dtype([None, 1], tf.int16)
        _test_dtype([None, 1], tf.int32)
        _test_dtype([None, 1], tf.int64)


class TestImplicit(tf.test.TestCase):
    def test_init(self):
        Implicit(implicit=tf.placeholder(tf.float32, [None, 1]))
        Implicit(implicit=tf.placeholder(tf.float32, [None, 1, 3]))
        Implicit(implicit=tf.placeholder(tf.float32, [None, 1]), value_shape=10)
        Implicit(implicit=tf.placeholder(tf.float32, [None, 1, 3]), value_shape=10)

    def test_shape(self):
        def _test_shape(batch_shape, value_shape):
            if value_shape is None:
                shape = batch_shape
            elif not isinstance(value_shape, (list, tuple)):
                shape = batch_shape + [value_shape]
            else:
                shape = batch_shape + value_shape
            implicit = tf.placeholder(tf.float32, shape)
            dist = Implicit(implicit, value_shape)

            if value_shape is None:
                value_shape = []
            elif not isinstance(value_shape, (list, tuple)):
                value_shape = [value_shape]

            # static
            self.assertEqual(dist.get_batch_shape().as_list(), batch_shape)
            self.assertEqual(dist.get_value_shape().as_list(), value_shape)

            # dynamic
            if None not in batch_shape:
                self.assertTrue(dist._batch_shape().dtype is tf.int32)
                with self.test_session(use_gpu=True):
                    self.assertEqual(dist._batch_shape().eval().tolist(), batch_shape)

            self.assertTrue(dist._value_shape().dtype is tf.int32)
            with self.test_session(use_gpu=True):
                self.assertEqual(dist._value_shape().eval().tolist(), value_shape)

        # No value shape
        _test_shape([None, 1], None)
        _test_shape([12, 1], None)
        _test_shape([None, 10, 15], None)
        _test_shape([12, 10, 15], None)
        # With value shape
        _test_shape([None, 1], 5)
        _test_shape([12, 1], 5)
        _test_shape([None, 10, 15], [5, 3])
        _test_shape([12, 10, 15], [5, 3])

    def test_sample(self):
        # with no value_shape
        implicit = Implicit(implicit=tf.placeholder(tf.float32, [None, 3]))
        with self.assertRaisesRegexp(
                ValueError, "ImplicitDistribution does not accept `n_samples` argument."):
            implicit.sample(3)
        with self.assertRaisesRegexp(
                ValueError, "ImplicitDistribution does not accept `n_samples` argument."):
            implicit.sample(tf.placeholder(tf.int32, None))

        utils.test_1parameter_sample_shape_same(self, Implicit, np.zeros, only_one_sample=True)

        # with value_shape
        implicit = Implicit(implicit=tf.placeholder(tf.float32, [None, 3]), value_shape=5)
        with self.assertRaisesRegexp(
                ValueError, "ImplicitDistribution does not accept `n_samples` argument."):
            implicit.sample(3)
        with self.assertRaisesRegexp(
                ValueError, "ImplicitDistribution does not accept `n_samples` argument."):
            implicit.sample(tf.placeholder(tf.int32, None))

        utils.test_1parameter_sample_shape_same(self, Implicit, np.zeros, only_one_sample=True)

    def test_log_prob_shape(self):
        def _make_param(shape):
            samples = np.zeros(shape)
            samples = samples.reshape((-1, shape[-1]))
            samples[:, 0] = 1
            return samples.reshape(shape)

        def _make_given(shape, dtype):
            samples = np.zeros(shape)
            samples = samples.reshape((-1, shape[-1]))
            samples[:, 0] = 1
            return samples.reshape(shape).astype(dtype)

        utils.test_1parameter_log_prob_shape_same(
            self, Implicit, _make_param, _make_given)

        def _distribution(param):
            return Implicit(param, 5)

        utils.test_1parameter_log_prob_shape_same(
            self, _distribution, _make_param, _make_given)

    def test_value(self):
        with self.test_session(use_gpu=True):
            def _test_value(implicit, given, value_shape, dtype):
                implicit = np.array(implicit, dtype)
                given = np.array(given, dtype)
                dist = Implicit(tf.constant(implicit), value_shape)
                p = dist.prob(tf.constant(given))
                log_p = dist.log_prob(tf.constant(given))
                prob = np.equal(implicit, given)
                if dtype == np.int32:
                    target_p = prob.astype(np.float32)
                else:
                    target_p = (2 * prob - 1) * np.inf
                target_log_p = np.log(target_p)
                self.assertAllClose(log_p.eval(), target_log_p)
                self.assertAllClose(p.eval(), target_p)

            _test_value([-50., -20., 0.], [-50., -20., 0.], None, np.float32)
            _test_value([-50., -20., 0.], [1, 0, 3], None, np.float32)
            _test_value([-50., -20., 0.], [-50., -20., 0.], None, np.int32)
            _test_value([-50., -20., 0.], [1, 0, 3], None, np.int32)
            _test_value([-50., -20., 0.], [-50., -20., 0.], 3, np.float32)
            _test_value([-50., -20., 0.], [1, 0, 3], 3, np.float32)
            _test_value([-50., -20., 0.], [-50., -20., 0.], 3, np.int32)
            _test_value([-50., -20., 0.], [1, 0, 3], 3, np.int32)
