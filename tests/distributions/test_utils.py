#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
from scipy import misc

from tests.context import zhusuan
from zhusuan.distributions.utils import *


class TestLogFactorial(tf.test.TestCase):
    def test_log_factorial(self):
        with self.test_session(use_gpu=True):
            for i in [0, 1, 2, 5, 10]:
                self.assertNear(np.log(misc.factorial(i)),
                                log_factorial(i).eval(), 1e-6)

            for i in [[2], [[1, 2], [3, 4]]]:
                self.assertAllClose(np.log(misc.factorial(i)),
                                    log_factorial(i).eval())


class TestLogCombination(tf.test.TestCase):
    def test_log_combination(self):
        with self.test_session(use_gpu=True):
            def _test_func(n, ks):
                true_value = np.log(misc.factorial(n)) - \
                    np.sum(np.log(misc.factorial(ks)), axis=-1)
                test_value = log_combination(n, ks).eval()
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
