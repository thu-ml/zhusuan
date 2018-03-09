#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from functools import partial
import tensorflow as tf
import numpy as np

from tests.distributions import utils
from zhusuan.distributions.special import *


class TestEmpirical(tf.test.TestCase):
    def test_init(self):
        Empirical(tf.float32, batch_shape=None, value_shape=None)
        Empirical(tf.float32, batch_shape=[None, 1])
        Empirical(tf.float32, batch_shape=[None, 10], value_shape=5)
        Empirical(tf.int32, batch_shape=[None, 1])
        Empirical(tf.int32, batch_shape=[None, 10], value_shape=5)

    def test_shape(self):
        def _test_shape(batch_shape, value_shape):
            empirical = Empirical(tf.float32, batch_shape, value_shape)

            # static
            self.assertEqual(empirical.get_batch_shape().as_list(),
                             batch_shape)
            if value_shape is None:
                self.assertEqual(empirical.get_value_shape(),
                                 tf.TensorShape(None))
            elif isinstance(value_shape, (list, tuple)):
                self.assertEqual(empirical.get_value_shape().as_list(),
                                 value_shape)
            else:
                self.assertEqual(empirical.get_value_shape().as_list(),
                                 [value_shape])

            # Error for _batch_shape:
            with self.assertRaises(NotImplementedError):
                empirical._batch_shape()

            # Error for _value_shape()
            with self.assertRaises(NotImplementedError):
                empirical._value_shape()

        # No value shape
        _test_shape([None, 1], None)
        _test_shape([12, 1], None)
        _test_shape([None, 10, 15], None)
        _test_shape([12, 10, 15], None)
        # With value shape
        _test_shape([None, 1], 5)
        _test_shape([12, 1], [])
        _test_shape([None, 10, 15], [5, 3])
        _test_shape([12, 10, 15], [])

    def test_sample(self):
        with self.assertRaisesRegexp(
                ValueError,
                "You can not sample from an Empirical distribution."):
            Empirical(tf.float32, batch_shape=[None, 1]).sample()

    def test_prob(self):
        with self.assertRaisesRegexp(
                ValueError,
                "An empirical distribution has no probability measure."):
            Empirical(tf.float32, batch_shape=[None, 1]).prob(
                np.zeros((5, 1)))

        with self.assertRaisesRegexp(
                ValueError,
                "An empirical distribution has no probability measure."):
            Empirical(tf.float32, batch_shape=[None, 1]).log_prob(
                np.zeros((5, 1)))

    def test_dtype(self):
        def _test_dtype(batch_shape, dtype):
            self.assertEqual(Empirical(dtype, batch_shape=batch_shape).dtype,
                             dtype)

        _test_dtype([None, 1], tf.float16)
        _test_dtype([None, 1], tf.float32)
        _test_dtype([None, 1], tf.float64)
        _test_dtype([None, 1], tf.int16)
        _test_dtype([None, 1], tf.int32)
        _test_dtype([None, 1], tf.int64)


class TestImplicit(tf.test.TestCase):
    def test_init(self):
        Implicit(samples=tf.placeholder(tf.float32, [None, 1]))
        Implicit(samples=tf.placeholder(tf.float32, [None, 1, 3]))
        Implicit(samples=tf.placeholder(tf.float32, [None, 1]),
                 value_shape=10)
        Implicit(samples=tf.placeholder(tf.float32, [None, 1, 3]),
                 value_shape=10)

    def test_shape(self):
        def _test_shape(batch_shape, value_shape):
            shape = tf.TensorShape(batch_shape).concatenate(
                tf.TensorShape(value_shape))
            implicit = tf.placeholder(tf.float32, shape)
            dist = Implicit(implicit, value_shape)

            # static
            if batch_shape is not None and value_shape is not None:
                self.assertEqual(dist.get_batch_shape().as_list(), batch_shape)
            if value_shape is not None:
                if not isinstance(value_shape, (list, tuple)):
                    value_shape = [value_shape]
                self.assertEqual(dist.get_value_shape().as_list(), value_shape)

            # dynamic
            with self.assertRaises(NotImplementedError):
                dist._batch_shape()
            with self.assertRaises(NotImplementedError):
                dist._value_shape()

        # No value shape
        _test_shape([None, 1], None)
        _test_shape([12, 1], None)
        _test_shape([None, 10, 15], None)
        _test_shape([12, 10, 15], None)
        # With value shape
        _test_shape([None, 1], [])
        _test_shape([None, 1], 5)
        _test_shape([12, 1], [])
        _test_shape([12, 1], 5)
        _test_shape([None, 10, 15], [])
        _test_shape([None, 10, 15], [5, 3])
        _test_shape([12, 10, 15], [])
        _test_shape([12, 10, 15], [5, 3])

    def test_sample(self):
        # with no value_shape
        implicit = Implicit(samples=tf.placeholder(tf.float32, [None, 3]))
        with self.assertRaisesRegexp(
                ValueError,
                "Implicit distribution does not accept `n_samples` argument."):
            implicit.sample(3)
        with self.assertRaisesRegexp(
                ValueError,
                "Implicit distribution does not accept `n_samples` argument."):
            implicit.sample(tf.placeholder(tf.int32, None))

        utils.test_1parameter_sample_shape_same(
            self, Implicit, np.zeros, only_one_sample=True)

        # with value_shape
        implicit = Implicit(samples=tf.placeholder(tf.float32, [None, 3]),
                            value_shape=5)
        with self.assertRaisesRegexp(
                ValueError,
                "Implicit distribution does not accept `n_samples` argument."):
            implicit.sample(3)
        with self.assertRaisesRegexp(
                ValueError,
                "Implicit distribution does not accept `n_samples` argument."):
            implicit.sample(tf.placeholder(tf.int32, None))

        utils.test_1parameter_sample_shape_same(self, Implicit, np.zeros,
                                                only_one_sample=True)

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
            self, partial(Implicit, value_shape=[]),
            _make_param, _make_given)

        utils.test_1parameter_log_prob_shape_same(
            self, partial(Implicit, value_shape=5),
            _make_param, _make_given)

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
