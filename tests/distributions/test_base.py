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
        super(Dist, self).__init__(tf.float32, is_continuous=True)

    def _event_shape(self):
        return tf.constant([3], dtype=tf.int32)

    def _get_event_shape(self):
        return tf.TensorShape([3])

    def _batch_shape(self):
        return tf.constant([2], dtype=tf.int32)

    def _get_batch_shape(self):
        return tf.TensorShape([2])

    def _sample(self, n_samples):
        return tf.ones([n_samples, 2, 3])

    def _log_prob(self, given):
        return tf.reduce_sum(tf.zeros_like(given), -1)

    def _prob(self, given):
        return tf.reduce_prod(tf.ones_like(given), -1)


class TestDistributions(tf.test.TestCase):
    def test_distributions(self):
        with self.test_session():
            dist = Dist()
            self.assertTrue(dist.is_continuous is True)
            self.assertEqual(dist.dtype, tf.float32)

            # shape
            evt_shape = dist.event_shape
            self.assertAllEqual(evt_shape.eval(), [3])
            static_evt_shape = dist.get_event_shape()
            self.assertAllEqual(static_evt_shape.as_list(), [3])

            btch_shape = dist.batch_shape
            self.assertAllEqual(btch_shape.eval(), [2])
            static_btch_shape = dist.get_batch_shape()
            self.assertAllEqual(static_btch_shape.as_list(), [2])

            # sample
            # static n_samples
            samples_1 = dist.sample()
            self.assertAllEqual(samples_1.eval(),
                                np.ones((2, 3), dtype=np.int32))
            samples_2 = dist.sample(n_samples=2)
            self.assertAllEqual(samples_2.eval(),
                                np.ones((2, 2, 3), dtype=np.int32))
            # dynamic n_samples
            n_samples = tf.placeholder(tf.int32, shape=[])
            samples_3 = dist.sample(n_samples=n_samples)
            self.assertAllEqual(samples_3.eval(feed_dict={n_samples: 1}),
                                np.ones((2, 3), dtype=np.int32))
            self.assertAllEqual(samples_3.eval(feed_dict={n_samples: 2}),
                                np.ones((2, 2, 3), dtype=np.int32))

            # log_prob
            given_1 = tf.ones([2, 3])
            log_p_1 = dist.log_prob(given_1)
            self.assertAllEqual(log_p_1.eval(), np.zeros((2)))

            given_2 = tf.ones([1, 2, 3])
            log_p_2 = dist.log_prob(given_2)
            self.assertAllEqual(log_p_2.eval(), np.zeros((1, 2)))

            given_3 = tf.placeholder(tf.float32, shape=None)
            log_p_3 = dist.log_prob(given_3)
            self.assertAllEqual(
                log_p_3.eval(feed_dict={given_3: np.ones((2, 3))}),
                np.zeros((2)))
            self.assertAllEqual(
                log_p_3.eval(feed_dict={given_3: np.ones((1, 2, 3))}),
                np.zeros((1, 2)))

            # prob
            p_1 = dist.prob(given_1)
            self.assertAllEqual(p_1.eval(), np.ones((2)))

            p_2 = dist.prob(given_2)
            self.assertAllEqual(p_2.eval(), np.ones((1, 2)))

            p_3 = dist.prob(given_3)
            self.assertAllEqual(
                p_3.eval(feed_dict={given_3: np.ones((2, 3))}),
                np.ones((2)))
            self.assertAllEqual(
                p_3.eval(feed_dict={given_3: np.ones((1, 2, 3))}),
                np.ones((1, 2)))
