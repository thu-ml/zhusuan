#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np

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
            self.assertEqual(norm.value_shape.eval().tolist(), [])

    def test_batch_shape(self):
        # static
        def _test_static(mean_shape, logstd_shape, target_shape,
                         group_event_ndims=0):
            mean = tf.placeholder(tf.float32, mean_shape)
            logstd = tf.placeholder(tf.float32, logstd_shape)
            norm = Normal(mean, logstd, group_event_ndims)
            self.assertEqual(norm.get_batch_shape().as_list(),
                             target_shape)

        _test_static([2, 3], [], [2, 3])
        _test_static([2, 3], [3], [2, 3])
        _test_static([2, 1, 4], [2, 3, 4], [2, 3, 4])
        _test_static([2, 3, 5], [3, 1], [2, 3, 5])
        _test_static([1, 2, 3], [1, 3], [1, 2, 3], group_event_ndims=2)
        _test_static([None, 3, 5], [3, None], [None, 3, 5])
        _test_static([2, None], [], [2, None])

        # dynamic
        with self.test_session(use_gpu=True):
            def _test_dynamic(mean_shape, logstd_shape, target_shape,
                              group_event_ndims=0):
                mean = tf.placeholder(tf.float32, None)
                logstd = tf.placeholder(tf.float32, None)
                norm = Normal(mean, logstd, group_event_ndims)
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

    def test_sample(self):
        pass

    def test_log_prob(self):
        pass

    def test_prob(self):
        pass


class TestBernoulli(tf.test.TestCase):
    pass


class TestCategorical(tf.test.TestCase):
    pass


class TestUniform(tf.test.TestCase):
    pass


class TestGamma(tf.test.TestCase):
    pass


class TestBeta(tf.test.TestCase):
    pass
