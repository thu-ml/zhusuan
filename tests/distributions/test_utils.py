#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
from scipy.special import factorial

from zhusuan.distributions.utils import *
from zhusuan.distributions.utils import get_shape_list, get_shape_at, \
    assert_rank_at_least


class TestLogCombination(tf.test.TestCase):
    def test_log_combination(self):
        with self.session(use_gpu=True):
            def _test_func(n, ks):
                tf_n = tf.convert_to_tensor(n, tf.float32)
                tf_ks = tf.convert_to_tensor(ks, tf.float32)
                true_value = np.log(factorial(n)) - \
                    np.sum(np.log(factorial(ks)), axis=-1)
                test_value = log_combination(tf_n, tf_ks).eval()
                self.assertAllClose(true_value, test_value)

            _test_func(10, [1, 2, 3, 4])
            _test_func([1, 2], [[1], [2]])
            _test_func([1, 4], [[1, 0], [2, 2]])
            _test_func([[2], [3]], [[[0, 2], [1, 2]]])


class TestExplicitBroadcast(tf.test.TestCase):
    def test_explicit_broadcast(self):
        with self.session(use_gpu=True):
            def _test_func(a_shape, b_shape, target_shape):
                a = tf.ones(a_shape)
                b = tf.ones(b_shape)
                a, b = explicit_broadcast(a, b, 'a', 'b')
                self.assertEqual(a.eval().shape, b.eval().shape)
                self.assertEqual(a.eval().shape, target_shape)

            _test_func((5, 4), (1,), (5, 4))
            _test_func((5, 4), (4,), (5, 4))
            _test_func((2, 3, 5), (2, 1, 5), (2, 3, 5))
            _test_func((2, 3, 5), (3, 5), (2, 3, 5))
            _test_func((2, 3, 5), (3, 1), (2, 3, 5))

            with self.assertRaisesRegexp(ValueError, "cannot broadcast"):
                _test_func((3,), (4,), None)
            with self.assertRaisesRegexp(ValueError, "cannot broadcast"):
                _test_func((2, 1), (2, 4, 3), None)


class TestIsSameDynamicShape(tf.test.TestCase):
    def test_is_same_dynamic_shape(self):
        with self.session(use_gpu=True):
            def _test(x_shape, y_shape, is_same):
                x = tf.ones(x_shape)
                y = tf.ones(y_shape)
                test_value = is_same_dynamic_shape(x, y)
                self.assertEqual(test_value.eval(), is_same)

            _test([1, 2], [1, 2], True)
            _test([2], [2, 2], False)
            _test([], [1], False)
            _test([1, 2], [2, 2], False)
            _test([], [], True)
            _test([3], [2], False)


class TestGetShapeList(tf.test.TestCase):
    def test_get_shape_list(self):
        with self.session(use_gpu=True):
            def test_shape_static(shape):
                ph = tf.placeholder(tf.float32, shape)
                self.assertEqual(get_shape_list(ph), shape)
            test_shape_static([2, 3])
            test_shape_static(None)
            # Dynamic
            ph = tf.placeholder(tf.float32, [2, None])
            fd = {ph: np.ones([2, 9])}
            shapes = get_shape_list(ph)
            self.assertEqual(shapes[0], 2)
            self.assertEqual(shapes[1].eval(fd), 9)


class TestGetShapeAt(tf.test.TestCase):
    def test_get_shape_at(self):
        with self.session(use_gpu=True):
            ph = tf.placeholder(tf.float32, [2, None])
            # Static
            self.assertEqual(get_shape_at(ph, 0), 2)
            # Dynamic
            fd = {ph: np.ones([2, 9])}
            self.assertEqual(get_shape_at(ph, 1).eval(fd), 9)


class TestAssertRankAtLeast(tf.test.TestCase):
    def test_assert_rank_at_least(self):
        with self.session(use_gpu=True):
            # Static
            ph = tf.placeholder(tf.float32, [2, None])
            assert_rank_at_least(ph, 2, 'ph')
            with self.assertRaisesRegexp(ValueError, 'should have rank'):
                assert_rank_at_least(ph, 3, 'ph')
            # Dynamic
            ph = tf.placeholder(tf.float32, None)
            assert_2 = assert_rank_at_least(ph, 2, 'ph')
            assert_3 = assert_rank_at_least(ph, 3, 'ph')
            fd = {ph: np.ones([2, 9])}
            assert_2.eval(fd)
            with self.assertRaises(tf.errors.InvalidArgumentError):
                assert_3.eval(fd)
