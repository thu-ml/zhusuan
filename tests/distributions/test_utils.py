#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
from scipy import misc

from zhusuan.distributions.utils import *


class TestLogCombination(tf.test.TestCase):
    def test_log_combination(self):
        with self.test_session(use_gpu=True):
            def _test_func(n, ks):
                tf_n = tf.convert_to_tensor(n, tf.float32)
                tf_ks = tf.convert_to_tensor(ks, tf.float32)
                true_value = np.log(misc.factorial(n)) - \
                    np.sum(np.log(misc.factorial(ks)), axis=-1)
                test_value = log_combination(tf_n, tf_ks).eval()
                self.assertAllClose(true_value, test_value)

            _test_func(10, [1, 2, 3, 4])
            _test_func([1, 2], [[1], [2]])
            _test_func([1, 4], [[1, 0], [2, 2]])
            _test_func([[2], [3]], [[[0, 2], [1, 2]]])


class TestExplicitBroadcast(tf.test.TestCase):
    def test_explicit_broadcast(self):
        with self.test_session(use_gpu=True):
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
        with self.test_session(use_gpu=True):
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
