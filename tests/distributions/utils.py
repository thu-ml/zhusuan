#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
from zhusuan.distributions.multivariate import Dirichlet


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
            regexp_msg = "must have the same dtype as"
        else:
            regexp_msg = "must have a dtype in"
        with test_class.assertRaisesRegexp(TypeError, regexp_msg):
            _test_parameter_dtype(None, param1_dtype, param2_dtype)

    _test_parameter_dtype_raise(tf.float16, tf.float32)
    _test_parameter_dtype_raise(tf.float32, tf.float64)
    _test_parameter_dtype_raise(tf.int32, tf.int32)


def test_dtype_1parameter_discrete(
        test_class, Distribution, prob_only=False, allow_16bit=True):
    def _test_sample_dtype(input_, result_dtype, **kwargs):
        distribution = Distribution(input_, **kwargs)
        samples = distribution.sample(2)
        test_class.assertEqual(distribution.dtype, result_dtype)
        test_class.assertEqual(samples.dtype, result_dtype)

    def _test_sample_dtype_raise(input_, dtype):
        with test_class.assertRaisesRegexp(TypeError,
                                           r"`dtype`.*not in"):
            _ = Distribution(input_, dtype=dtype)

    if not prob_only:
        for input_ in [[1.], [[2., 3.], [4., 5.]]]:
            _test_sample_dtype(input_, tf.int32)

            if allow_16bit:
                _test_sample_dtype(input_, tf.int16, dtype=tf.int16)
                _test_sample_dtype(input_, tf.float16, dtype=tf.float16)
            else:
                _test_sample_dtype_raise(input_, dtype=tf.int16)
                _test_sample_dtype_raise(input_, dtype=tf.float16)

            _test_sample_dtype(input_, tf.int32, dtype=tf.int32)
            _test_sample_dtype(input_, tf.int64, dtype=tf.int64)
            _test_sample_dtype(input_, tf.float32, dtype=tf.float32)
            _test_sample_dtype(input_, tf.float64, dtype=tf.float64)
            _test_sample_dtype_raise(input_, dtype=tf.uint8)
            _test_sample_dtype_raise(input_, dtype=tf.bool)

    def _test_parameter_dtype_raise(param_dtype):
        param = tf.placeholder(param_dtype, [1])
        with test_class.assertRaisesRegexp(TypeError,
                                           "must have a dtype in"):
            Distribution(param)

    if not allow_16bit:
        _test_parameter_dtype_raise(tf.float16)

    _test_parameter_dtype_raise(tf.uint8)
    _test_parameter_dtype_raise(tf.bool)
    _test_parameter_dtype_raise(tf.int16)
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

    if allow_16bit:
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


def test_batch_shape_2parameter_univariate(
        test_class, Distribution, make_param1, make_param2):
    # static
    def _test_static(param1_shape, param2_shape, target_shape):
        param1 = tf.placeholder(tf.float32, param1_shape)
        param2 = tf.placeholder(tf.float32, param2_shape)
        dist = Distribution(param1, param2)
        if dist.get_batch_shape():
            test_class.assertEqual(dist.get_batch_shape().as_list(),
                                   target_shape)
        else:
            test_class.assertEqual(None, target_shape)

    _test_static([2, 3], [], [2, 3])
    _test_static([2, 3], [3], [2, 3])
    _test_static([2, 1, 4], [2, 3, 4], [2, 3, 4])
    _test_static([2, 3, 5], [3, 1], [2, 3, 5])
    _test_static([1, 2, 3], [1, 3], [1, 2, 3])
    _test_static([None, 3, 5], [3, None], [None, 3, 5])
    _test_static([None, 1, 3], [None, 1], [None, None, 3])
    _test_static([None, 1, 10], [None, 1, 10], [None, 1, 10])
    _test_static([2, None], [], [2, None])
    _test_static(None, [1, 2], None)

    # dynamic
    with test_class.session(use_gpu=True):
        def _test_dynamic(param1_shape, param2_shape, target_shape):
            param1 = tf.placeholder(tf.float32, None)
            param2 = tf.placeholder(tf.float32, None)
            dist = Distribution(param1, param2)
            test_class.assertTrue(dist.batch_shape.dtype is tf.int32)
            test_class.assertEqual(
                dist.batch_shape.eval(
                    feed_dict={param1: make_param1(param1_shape),
                               param2: make_param2(param2_shape)}).tolist(),
                target_shape)

        _test_dynamic([2, 3], [], [2, 3])
        _test_dynamic([2, 3], [3], [2, 3])
        _test_dynamic([2, 1, 4], [2, 3, 4], [2, 3, 4])
        _test_dynamic([2, 3, 5], [3, 1], [2, 3, 5])
        with test_class.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                           "Incompatible shapes"):
            _test_dynamic([2, 3, 5], [3, 2], None)


def test_2parameter_sample_shape_same(
        test_class, Distribution, make_param1, make_param2):
    def _test_static(param1_shape, param2_shape, n_samples, target_shape):
        param1 = tf.placeholder(tf.float32, param1_shape)
        param2 = tf.placeholder(tf.float32, param2_shape)
        dist = Distribution(param1, param2)
        samples = dist.sample(n_samples)
        if samples.get_shape():
            test_class.assertEqual(samples.get_shape().as_list(), target_shape)
        else:
            test_class.assertEqual(None, target_shape)

    _test_static([2, 3], [], None, [2, 3])
    _test_static([2, 3], [], 1, [1, 2, 3])
    _test_static([5], [5], 2, [2, 5])
    _test_static([2, 1, 4], [1, 2, 4], 3, [3, 2, 2, 4])
    _test_static([None, 2], [3, None], tf.placeholder(tf.int32, []),
                 [None, 3, 2])
    _test_static(None, [1, 2], None, None)
    _test_static(None, [1, 2], 1, None)
    _test_static([None, 1, 10], [None, 1, 10], None, [None, 1, 10])
    _test_static([3, None], [3, 1], 2, [2, 3, None])

    with test_class.session(use_gpu=True):
        def _test_dynamic(param1_shape, param2_shape, n_samples,
                          target_shape):
            param1 = tf.placeholder(tf.float32, None)
            param2 = tf.placeholder(tf.float32, None)
            dist = Distribution(param1, param2)
            samples = dist.sample(n_samples)
            test_class.assertEqual(
                tf.shape(samples).eval(
                    feed_dict={param1: make_param1(param1_shape),
                               param2: make_param2(param2_shape)}).tolist(),
                target_shape)

        _test_dynamic([2, 3], [2, 1], 1, [1, 2, 3])
        _test_dynamic([1, 3], [], 2, [2, 1, 3])
        _test_dynamic([2, 1, 5], [3, 1], 3, [3, 2, 3, 5])
        with test_class.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                           "Incompatible shapes"):
            _test_dynamic([2, 3, 5], [2, 1], 1, None)


def test_2parameter_log_prob_shape_same(
        test_class, Distribution, make_param1, make_param2, make_given):
    def _test_static(param1_shape, param2_shape, given_shape, target_shape):
        param1 = tf.placeholder(tf.float32, param1_shape)
        param2 = tf.placeholder(tf.float32, param2_shape)
        given = tf.placeholder(tf.float32, given_shape)
        dist = Distribution(param1, param2)
        log_p = dist.log_prob(given)
        if log_p.get_shape():
            test_class.assertEqual(log_p.get_shape().as_list(), target_shape)
        else:
            test_class.assertEqual(None, target_shape)

    _test_static([2, 3], [], [2, 3], [2, 3])
    _test_static([5], [5], [2, 1], [2, 5])
    _test_static([None, 2], [3, None], [None, 1, 1], [None, 3, 2])
    _test_static(None, [1, 2], [2, 2], None)
    _test_static([3, None], [3, 1], [3, 2, 1, 1], [3, 2, 3, None])

    with test_class.session(use_gpu=True):
        def _test_dynamic(param1_shape, param2_shape, given_shape,
                          target_shape):
            param1 = tf.placeholder(tf.float32, None)
            param2 = tf.placeholder(tf.float32, None)
            dist = Distribution(param1, param2)
            given = tf.placeholder(tf.float32, None)
            log_p = dist.log_prob(given)
            test_class.assertEqual(
                tf.shape(log_p).eval(
                    feed_dict={param1: make_param1(param1_shape),
                               param2: make_param2(param2_shape),
                               given: make_given(given_shape)}).tolist(),
                target_shape)

        _test_dynamic([2, 3], [2, 1], [1, 3], [2, 3])
        _test_dynamic([1, 3], [], [2, 1, 3], [2, 1, 3])
        _test_dynamic([1, 5], [3, 1], [1, 2, 1, 1], [1, 2, 3, 5])
        with test_class.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                           "Incompatible shapes"):
            _test_dynamic([2, 3, 5], [], [1, 2, 1], None)


def test_batch_shape_1parameter(
        test_class, Distribution, make_param, is_univariate):
    # static
    def _test_static(param_shape):
        param = tf.placeholder(tf.float32, param_shape)
        dist = Distribution(param)
        if dist.get_batch_shape():
            if not is_univariate:
                param_shape = param_shape[:-1]
            test_class.assertEqual(dist.get_batch_shape().as_list(),
                                   param_shape)
        else:
            test_class.assertEqual(None, param_shape)

    if is_univariate:
        _test_static([])
    _test_static([2])
    _test_static([2, 3])
    _test_static([2, 1, 4])
    _test_static([None])
    _test_static([None, 3, 5])
    _test_static([1, None, 3])
    _test_static(None)

    # dynamic
    with test_class.session(use_gpu=True):
        def _test_dynamic(param_shape):
            param = tf.placeholder(tf.float32, None)
            dist = Distribution(param)
            test_class.assertTrue(dist.batch_shape.dtype is tf.int32)
            test_class.assertEqual(
                dist.batch_shape.eval(
                    feed_dict={param: make_param(param_shape)}).tolist(),
                param_shape if is_univariate else param_shape[:-1])

        if is_univariate:
            _test_dynamic([])
        _test_dynamic([2])
        _test_dynamic([2, 3])
        _test_dynamic([2, 1, 4])


def test_1parameter_sample_shape_same(
        test_class, Distribution, make_param, only_one_sample=False):
    def _test_static(param_shape, n_samples, target_shape):
        param = tf.placeholder(tf.float32, param_shape)
        dist = Distribution(param)
        samples = dist.sample(n_samples)
        if samples.get_shape():
            test_class.assertEqual(samples.get_shape().as_list(), target_shape)
        else:
            test_class.assertEqual(None, target_shape)

    _test_static([2, 3], None, [2, 3])
    _test_static([2, 3], 1, [1, 2, 3])
    if not only_one_sample:
        _test_static([5], 2, [2, 5])
        _test_static([None, 2], tf.placeholder(tf.int32, []), [None, None, 2])
    _test_static(None, 1, None)
    _test_static(None, None, None)
    _test_static([None, 1, 10], None, [None, 1, 10])
    if not only_one_sample:
        _test_static([3, None], 2, [2, 3, None])

    with test_class.session(use_gpu=True):
        def _test_dynamic(param_shape, n_samples, target_shape):
            param = tf.placeholder(tf.float32, None)
            dist = Distribution(param)
            samples = dist.sample(n_samples)
            test_class.assertEqual(
                tf.shape(samples).eval(
                    feed_dict={param: make_param(param_shape)}).tolist(),
                target_shape)

        _test_dynamic([2, 3], 1, [1, 2, 3])
        if not only_one_sample:
            _test_dynamic([1, 3], 2, [2, 1, 3])
            _test_dynamic([2, 1, 5], 3, [3, 2, 1, 5])


def test_1parameter_log_prob_shape_same(
        test_class, Distribution, make_param, make_given):
    def _test_static(param_shape, given_shape, target_shape):
        param = tf.placeholder(tf.float32, param_shape)
        dist = Distribution(param)
        given = tf.placeholder(dist.dtype, given_shape)
        log_p = dist.log_prob(given)
        if log_p.get_shape():
            test_class.assertEqual(log_p.get_shape().as_list(), target_shape)
        else:
            test_class.assertEqual(None, target_shape)

    _test_static([2, 3], [2, 1], [2, 3])
    _test_static([5], [2, 1], [2, 5])
    _test_static([None, 2], [3, None], [3, 2])
    _test_static([None, 2], [None, 1, 1], [None, None, 2])
    _test_static(None, [2, 2], None)
    _test_static([3, None], [3, 2, 1, 1], [3, 2, 3, None])
    with test_class.assertRaisesRegexp(ValueError, "broadcast to match"):
        _test_static([2, 3, 5], [1, 2, 1], None)

    with test_class.session(use_gpu=True):
        def _test_dynamic(param_shape, given_shape, target_shape):
            param = tf.placeholder(tf.float32, None)
            dist = Distribution(param)
            given = tf.placeholder(dist.dtype, None)
            log_p = dist.log_prob(given)
            numpy_given_dtype = dist.dtype.as_numpy_dtype
            test_class.assertEqual(
                tf.shape(log_p).eval(
                    feed_dict={param: make_param(param_shape),
                               given: make_given(given_shape,
                                                 numpy_given_dtype)}).tolist(),
                target_shape)

        _test_dynamic([2, 3], [1, 3], [2, 3])
        _test_dynamic([1, 3], [2, 2, 3], [2, 2, 3])
        _test_dynamic([1, 5], [1, 2, 3, 1], [1, 2, 3, 5])
        with test_class.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                           "Incompatible shapes"):
            _test_dynamic([2, 3, 5], [1, 2, 1], None)


def test_1parameter_sample_shape_one_rank_less(
        test_class, Distribution, make_param):
    def _test_static(param_shape, n_samples, target_shape):
        param = tf.placeholder(tf.float32, param_shape)
        dist = Distribution(param)
        samples = dist.sample(n_samples)
        if samples.get_shape():
            test_class.assertEqual(samples.get_shape().as_list(), target_shape)
        else:
            test_class.assertEqual(None, target_shape)

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

    with test_class.session(use_gpu=True):
        def _test_dynamic(param_shape, n_samples, target_shape):
            param = tf.placeholder(tf.float32, None)
            dist = Distribution(param)
            samples = dist.sample(n_samples)
            test_class.assertEqual(
                tf.shape(samples).eval(
                    feed_dict={param: make_param(param_shape)}).tolist(),
                target_shape)

        _test_dynamic([2], 1, [1, 2])
        _test_dynamic([2, 3], 1, [1, 2, 3])
        _test_dynamic([1, 3], 2, [2, 1, 3])
        _test_dynamic([2, 1, 5], 3, [3, 2, 1, 5])


def test_1parameter_log_prob_shape_one_rank_less(
        test_class, Distribution, make_param, make_given):
    def _test_static(param_shape, given_shape, target_shape):
        param = tf.placeholder(tf.float32, param_shape)
        dist = Distribution(param)
        given = tf.placeholder(dist.dtype, given_shape)
        log_p = dist.log_prob(given)
        if log_p.get_shape():
            test_class.assertEqual(log_p.get_shape().as_list(), target_shape)
        else:
            test_class.assertEqual(None, target_shape)

    _test_static([2, 3], [2, 3], [2])
    _test_static([2, 5], [5], [2])
    _test_static([1, 2, 4], [4], [1, 2])
    _test_static([3, 1, 5], [1, 4, 5], [3, 4])
    _test_static([1, 4], [2, 5, 4], [2, 5])
    _test_static([None, 2, 4], [3, None, 4], [3, 2])
    _test_static([None, 2], [None, 1, 1, 2], [None, 1, None])
    _test_static(None, [2, 2], None)
    if Distribution != Dirichlet:
        # TODO: This failed with a bug in Tensorflow in Dirichlet.
        # https://github.com/tensorflow/tensorflow/issues/8391
        _test_static([3, None], [3, 2, 1, None], [3, 2, 3])
        _test_static([3, None], [3, 2, 1, 1], [3, 2, 3])
    with test_class.assertRaisesRegexp(ValueError, "broadcast to match"):
        _test_static([2, 3, 5], [1, 2, 5], None)

    with test_class.session(use_gpu=True):
        def _test_dynamic(param_shape, given_shape, target_shape):
            param = tf.placeholder(tf.float32, None)
            dist = Distribution(param)
            given = tf.placeholder(dist.dtype, None)
            log_p = dist.log_prob(given)

            test_class.assertEqual(
                tf.shape(log_p).eval(
                    feed_dict={param: make_param(param_shape),
                               given: make_given(given_shape)}).tolist(),
                target_shape)

        _test_dynamic([2, 3, 3], [1, 3], [2, 3])
        _test_dynamic([1, 3], [2, 2, 3], [2, 2])
        _test_dynamic([1, 5, 2], [1, 2, 1, 1], [1, 2, 5])
        if Distribution != Dirichlet:
            _test_dynamic([1, 5, 1], [1, 2, 1, 1], [1, 2, 5])
        with test_class.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                           "Incompatible shapes"):
            _test_dynamic([2, 3, 5], [1, 2, 5], None)
