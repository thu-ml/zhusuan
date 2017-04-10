#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np


def test_dtype_2parameter(test_class, Distribution):
    # Test sample dtype
    def _test_sample_dtype(dtype):
        param1 = tf.placeholder(dtype, None)
        param2 = tf.placeholder(dtype, None)
        distribution = Distribution(param1, param2)
        test_class.assertEqual(dtype, distribution.sample(1).dtype)

    _test_sample_dtype(tf.float16)
    _test_sample_dtype(tf.float32)
    _test_sample_dtype(tf.float64)

    # Test log_prob and prob dtype
    def _test_log_prob_dtype(dtype):
        param1 = tf.placeholder(dtype, None)
        param2 = tf.placeholder(dtype, None)
        distribution = Distribution(param1, param2)

        # test for tensor
        given = tf.placeholder(dtype, None)
        test_class.assertEqual(distribution.prob(given).dtype, dtype)
        test_class.assertEqual(distribution.log_prob(given).dtype, dtype)

        # test for numpy
        given_np = np.array([1], dtype.as_numpy_dtype)
        test_class.assertEqual(distribution.prob(given_np).dtype, dtype)
        test_class.assertEqual(distribution.log_prob(given_np).dtype, dtype)

    _test_log_prob_dtype(tf.float16)
    _test_log_prob_dtype(tf.float32)
    _test_log_prob_dtype(tf.float64)

    # Test dtype for parameters
    def _test_parameter_dtype(result_dtype, param1_dtype, param2_dtype):
        param1 = tf.placeholder(param1_dtype, None)
        param2 = tf.placeholder(param2_dtype, None)
        distribution = Distribution(param1, param2)
        test_class.assertEqual(distribution.dtype, result_dtype)

    _test_parameter_dtype(tf.float16, tf.float16, tf.float16)
    _test_parameter_dtype(tf.float32, tf.float32, tf.float32)
    _test_parameter_dtype(tf.float64, tf.float64, tf.float64)

    def _test_parameter_dtype_raise(param1_dtype, param2_dtype):
        if param1_dtype != param2_dtype:
            regexp_msg = "must be the same type as"
        else:
            regexp_msg = "must be in"
        with test_class.assertRaisesRegexp(TypeError, regexp_msg):
            _test_parameter_dtype(None, param1_dtype, param2_dtype)

    _test_parameter_dtype_raise(tf.float16, tf.float32)
    _test_parameter_dtype_raise(tf.float32, tf.float64)
    _test_parameter_dtype_raise(tf.int32, tf.int32)


def test_dtype_1parameter_discrete(test_class, Distribution):
    def _test_sample_dtype(result_dtype, dtype):
        distribution = Distribution([1.], dtype=dtype)
        samples = distribution.sample(2)
        test_class.assertEqual(distribution.dtype, result_dtype)
        test_class.assertEqual(samples.dtype, result_dtype)

    _test_sample_dtype(tf.int32, None)
    _test_sample_dtype(tf.int16, tf.int16)
    _test_sample_dtype(tf.int32, tf.int32)
    _test_sample_dtype(tf.float32, tf.float32)
    _test_sample_dtype(tf.float64, tf.float64)

    def _test_parameter_dtype_raise(param_dtype):
        param = tf.placeholder(param_dtype, [1])
        with test_class.assertRaises(TypeError):
            Distribution(param)

    _test_parameter_dtype_raise(tf.int32)
    _test_parameter_dtype_raise(tf.int64)

    # test dtype for log_prob and prob
    def _test_log_prob_dtype(param_dtype, given_dtype):
        param = tf.placeholder(param_dtype, [1])
        distribution = Distribution(param, dtype=given_dtype)
        test_class.assertEqual(distribution.param_dtype, param_dtype)

        # test for tensor
        given = tf.placeholder(given_dtype, None)
        prob = distribution.prob(given)
        log_prob = distribution.log_prob(given)

        test_class.assertEqual(prob.dtype, param_dtype)
        test_class.assertEqual(log_prob.dtype, param_dtype)

        # test for numpy
        given_np = np.array([1], given_dtype.as_numpy_dtype)
        prob_np = distribution.prob(given_np)
        log_prob_np = distribution.log_prob(given_np)

        test_class.assertEqual(prob_np.dtype, param_dtype)
        test_class.assertEqual(log_prob_np.dtype, param_dtype)

    _test_log_prob_dtype(tf.float16, tf.int32)
    _test_log_prob_dtype(tf.float32, tf.int32)
    _test_log_prob_dtype(tf.float64, tf.int64)
    _test_log_prob_dtype(tf.float32, tf.float32)
    _test_log_prob_dtype(tf.float32, tf.float64)


def test_dtype_1parameter_continuous(test_class, Distribution):
    def _test_sample_dtype(dtype):
        param = tf.placeholder(dtype, [2])
        distribution = Distribution(param)
        samples = distribution.sample(2)
        test_class.assertEqual(distribution.dtype, dtype)
        test_class.assertEqual(samples.dtype, dtype)

    _test_sample_dtype(tf.float16)
    _test_sample_dtype(tf.float32)
    _test_sample_dtype(tf.float64)

    def _test_parameter_dtype_raise(param_dtype):
        param = tf.placeholder(param_dtype, [2])
        with test_class.assertRaises(TypeError):
            Distribution(param)

    _test_parameter_dtype_raise(tf.int32)
    _test_parameter_dtype_raise(tf.int64)

    # test dtype for log_prob and prob
    def _test_log_prob_dtype(dtype):
        param = tf.placeholder(dtype, [2])
        distribution = Distribution(param)

        # test for tensor
        given = tf.placeholder(dtype, None)
        test_class.assertEqual(distribution.prob(given).dtype, dtype)
        test_class.assertEqual(distribution.log_prob(given).dtype, dtype)

        # test for numpy
        given_np = np.array([1], dtype.as_numpy_dtype)
        test_class.assertEqual(distribution.prob(given_np).dtype, dtype)
        test_class.assertEqual(distribution.log_prob(given_np).dtype, dtype)

    _test_log_prob_dtype(tf.float16)
    _test_log_prob_dtype(tf.float32)
    _test_log_prob_dtype(tf.float64)
