#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np

from zhusuan.distributions.base import *


class Dist(Distribution):
    def __init__(self,
                 dtype=tf.float32,
                 param_dtype=tf.float32,
                 group_ndims=0,
                 shape_fully_defined=True,
                 **kwargs):
        super(Dist, self).__init__(dtype,
                                   param_dtype,
                                   is_continuous=True,
                                   is_reparameterized=True,
                                   group_ndims=group_ndims,
                                   **kwargs)
        self._shape_fully_defined = shape_fully_defined

    def _value_shape(self):
        return tf.constant([5], dtype=tf.int32)

    def _get_value_shape(self):
        if self._shape_fully_defined:
            return tf.TensorShape([5])
        return tf.TensorShape(None)

    def _batch_shape(self):
        return tf.constant([2, 3, 4], dtype=tf.int32)

    def _get_batch_shape(self):
        if self._shape_fully_defined:
            return tf.TensorShape([2, 3, 4])
        return tf.TensorShape([None, 3, 4])

    def _sample(self, n_samples):
        return tf.ones([n_samples, 2, 3, 4, 5])

    def _log_prob(self, given):
        return tf.reduce_sum(tf.zeros_like(given), -1)

    def _prob(self, given):
        return tf.reduce_prod(tf.ones_like(given), -1)


class TestDistributions(tf.test.TestCase):
    def test_baseclass(self):
        dist = Distribution(tf.float32,
                            param_dtype=tf.float32,
                            is_continuous=True,
                            is_reparameterized=True,
                            group_ndims=2)
        self.assertEqual(dist.dtype, tf.float32)
        self.assertEqual(dist.param_dtype, tf.float32)
        self.assertEqual(dist.is_continuous, True)
        self.assertEqual(dist.is_reparameterized, True)
        self.assertEqual(dist.group_ndims, 2)
        with self.assertRaises(NotImplementedError):
            dist._value_shape()
        with self.assertRaises(NotImplementedError):
            dist._get_value_shape()
        with self.assertRaises(NotImplementedError):
            dist._batch_shape()
        with self.assertRaises(NotImplementedError):
            dist._get_batch_shape()
        with self.assertRaises(NotImplementedError):
            dist._sample(n_samples=1)
        with self.assertRaises(NotImplementedError):
            dist._log_prob(tf.ones([2, 3, 4, 5]))
        with self.assertRaises(NotImplementedError):
            dist._prob(tf.ones([2, 3, 4, 5]))

        with self.assertRaisesRegexp(ValueError, "must be non-negative"):
            dist2 = Distribution(tf.float32, tf.float32, True, True, False, -1)

    def test_subclass(self):
        with self.session(use_gpu=True):
            dist = Dist(group_ndims=2)
            self.assertEqual(dist.dtype, tf.float32)
            self.assertEqual(dist.is_continuous, True)
            self.assertEqual(dist.is_reparameterized, True)
            self.assertEqual(dist.group_ndims, 2)

            # shape
            static_v_shape = dist.get_value_shape()
            self.assertAllEqual(static_v_shape.as_list(), [5])
            v_shape = dist.value_shape
            self.assertAllEqual(v_shape.eval(), [5])

            static_b_shape = dist.get_batch_shape()
            self.assertAllEqual(static_b_shape.as_list(), [2, 3, 4])
            b_shape = dist.batch_shape
            self.assertAllEqual(b_shape.eval(), [2, 3, 4])

            # sample
            # static n_samples
            samples_1 = dist.sample()
            self.assertAllEqual(samples_1.eval(),
                                np.ones((2, 3, 4, 5), dtype=np.int32))
            for n in [1, 2]:
                samples_2 = dist.sample(n_samples=n)
                self.assertAllEqual(samples_2.eval(),
                                    np.ones((n, 2, 3, 4, 5), dtype=np.int32))
            # dynamic n_samples
            n_samples = tf.placeholder(tf.int32)
            samples_3 = dist.sample(n_samples=n_samples)
            for n in [1, 2]:
                self.assertAllEqual(samples_3.eval(feed_dict={n_samples: n}),
                                    np.ones((n, 2, 3, 4, 5), dtype=np.int32))
            with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                         "should be a scalar"):
                samples_3.eval(feed_dict={n_samples: [1, 2]})

            # log_prob
            given_1 = tf.ones([2, 3, 4, 5])
            log_p_1 = dist.log_prob(given_1)
            self.assertAllEqual(log_p_1.eval(), np.zeros((2)))
            with self.assertRaisesRegexp(
                    ValueError,
                    r"broadcast to match batch_shape \+ value_shape"):
                dist.log_prob(tf.ones([3, 3, 4, 5]))

            given_2 = tf.ones([1, 2, 3, 4, 5])
            log_p_2 = dist.log_prob(given_2)
            self.assertAllEqual(log_p_2.eval(), np.zeros((1, 2)))

            given_3 = tf.placeholder(tf.float32, shape=None)
            log_p_3 = dist.log_prob(given_3)
            self.assertAllEqual(
                log_p_3.eval(feed_dict={given_3: np.ones((2, 3, 4, 5))}),
                np.zeros((2)))
            self.assertAllEqual(
                log_p_3.eval(feed_dict={given_3: np.ones((1, 1, 2, 3, 4, 5))}),
                np.zeros((1, 1, 2)))

            # prob
            p_1 = dist.prob(given_1)
            self.assertAllEqual(p_1.eval(), np.ones((2)))

            p_2 = dist.prob(given_2)
            self.assertAllEqual(p_2.eval(), np.ones((1, 2)))

            p_3 = dist.prob(given_3)
            self.assertAllEqual(
                p_3.eval(feed_dict={given_3: np.ones((2, 3, 4, 5))}),
                np.ones((2)))
            self.assertAllEqual(
                p_3.eval(feed_dict={given_3: np.ones((1, 1, 2, 3, 4, 5))}),
                np.ones((1, 1, 2)))

            with self.assertRaisesRegexp(ValueError, "has been deprecated"):
                Dist(group_event_ndims=1)

            group_ndims = tf.placeholder(tf.int32)
            dist2 = Dist(group_ndims=group_ndims)
            with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                         "must be non-negative"):
                dist2.group_ndims.eval(feed_dict={group_ndims: -1})
            with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                         "should be a scalar"):
                dist2.group_ndims.eval(feed_dict={group_ndims: [1, 2]})

            # shape not fully defined
            dist3 = Dist(shape_fully_defined=False)

            static_v_shape = dist3.get_value_shape()
            self.assertAllEqual(static_v_shape, tf.TensorShape(None))
            v_shape = dist3.value_shape
            self.assertAllEqual(v_shape.eval(), [5])

            static_b_shape = dist3.get_batch_shape()
            self.assertAllEqual(static_b_shape.as_list(), [None, 3, 4])
            b_shape = dist3.batch_shape
            self.assertAllEqual(b_shape.eval(), [2, 3, 4])

            # given type of log_prob and prob
            def _test_log_prob_raise(dtype, given_dtype):
                dist = Dist(dtype=dtype)

                given = tf.placeholder(given_dtype, None)
                with self.assertRaises(ValueError):
                    dist.prob(given)

                with self.assertRaises(ValueError):
                    dist.log_prob(given)

            _test_log_prob_raise(tf.float32, tf.float64)
            _test_log_prob_raise(tf.float32, tf.float16)
            _test_log_prob_raise(tf.float32, tf.int32)
            _test_log_prob_raise(tf.float32, tf.int64)
            _test_log_prob_raise(tf.float64, tf.float32)
            _test_log_prob_raise(tf.float64, tf.int32)
            _test_log_prob_raise(tf.int32, tf.float32)
            _test_log_prob_raise(tf.int32, tf.int64)
            _test_log_prob_raise(tf.int64, tf.int32)
            _test_log_prob_raise(tf.int64, tf.float64)
