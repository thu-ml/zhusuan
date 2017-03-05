#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np

from tests.context import zhusuan
from zhusuan.distributions.base import *


class Dist(Distribution):
    def __init__(self):
        super(Dist, self).__init__(tf.float32,
                                   is_continuous=True,
                                   is_reparameterized=True,
                                   group_event_ndims=2)

    def _value_shape(self):
        return tf.constant([5], dtype=tf.int32)

    def _get_value_shape(self):
        return tf.TensorShape([5])

    def _batch_shape(self):
        return tf.constant([2, 3, 4], dtype=tf.int32)

    def _get_batch_shape(self):
        return tf.TensorShape([2, 3, 4])

    def _sample(self, n_samples):
        return tf.ones([n_samples, 2, 3, 4, 5])

    def _log_prob(self, given):
        return tf.reduce_sum(tf.zeros_like(given), -1)

    def _prob(self, given):
        return tf.reduce_prod(tf.ones_like(given), -1)


class TestDistributions(tf.test.TestCase):
    def test_baseclass(self):
        dist = Distribution(tf.float32,
                            is_continuous=True,
                            is_reparameterized=True,
                            group_event_ndims=2)
        self.assertEqual(dist.dtype, tf.float32)
        self.assertEqual(dist.is_continuous, True)
        self.assertEqual(dist.is_reparameterized, True)
        self.assertEqual(dist.group_event_ndims, 2)
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

    def test_subclass(self):
        with self.test_session(use_gpu=True):
            dist = Dist()
            self.assertEqual(dist.dtype, tf.float32)
            self.assertEqual(dist.is_continuous, True)
            self.assertEqual(dist.is_reparameterized, True)
            self.assertEqual(dist.group_event_ndims, 2)

            # shape
            v_shape = dist.value_shape
            self.assertAllEqual(v_shape.eval(), [5])
            static_v_shape = dist.get_value_shape()
            self.assertAllEqual(static_v_shape.as_list(), [5])

            b_shape = dist.batch_shape
            self.assertAllEqual(b_shape.eval(), [2, 3, 4])
            static_b_shape = dist.get_batch_shape()
            self.assertAllEqual(static_b_shape.as_list(), [2, 3, 4])

            # sample
            # static n_samples
            samples_1 = dist.sample()
            self.assertAllEqual(samples_1.eval(),
                                np.ones((2, 3, 4, 5), dtype=np.int32))
            samples_2 = dist.sample(n_samples=2)
            self.assertAllEqual(samples_2.eval(),
                                np.ones((2, 2, 3, 4, 5), dtype=np.int32))
            # dynamic n_samples
            n_samples = tf.placeholder(tf.int32, shape=[])
            samples_3 = dist.sample(n_samples=n_samples)
            self.assertAllEqual(samples_3.eval(feed_dict={n_samples: 1}),
                                np.ones((2, 3, 4, 5), dtype=np.int32))
            self.assertAllEqual(samples_3.eval(feed_dict={n_samples: 2}),
                                np.ones((2, 2, 3, 4, 5), dtype=np.int32))

            # log_prob
            given_1 = tf.ones([2, 3, 4, 5])
            log_p_1 = dist.log_prob(given_1)
            self.assertAllEqual(log_p_1.eval(), np.zeros((2)))

            given_2 = tf.ones([1, 2, 3, 4, 5])
            log_p_2 = dist.log_prob(given_2)
            self.assertAllEqual(log_p_2.eval(), np.zeros((1, 2)))

            given_3 = tf.placeholder(tf.float32, shape=None)
            log_p_3 = dist.log_prob(given_3)
            self.assertAllEqual(
                log_p_3.eval(feed_dict={given_3: np.ones((2, 3, 4, 5))}),
                np.zeros((2)))
            self.assertAllEqual(
                log_p_3.eval(feed_dict={given_3: np.ones((1, 2, 3, 4, 5))}),
                np.zeros((1, 2)))

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
                p_3.eval(feed_dict={given_3: np.ones((1, 2, 3, 4, 5))}),
                np.ones((1, 2)))
