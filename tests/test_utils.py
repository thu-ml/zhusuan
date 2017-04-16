#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
from scipy import misc
import numpy as np

from zhusuan.utils import *
from zhusuan.utils import add_name_scope, if_raise, log_sum_exp


class TestLogSumExp(tf.test.TestCase):
    def test_log_sum_exp(self):
        with self.test_session(use_gpu=True) as sess:
            a = np.array([[[1., 3., 0.2], [0.7, 2., 1e-6]],
                          [[0., 1e6, 1.], [1., 1., 1.]]])
            for keepdims in [True, False]:
                true_values = misc.logsumexp(a, (0, 2), keepdims=keepdims)
                test_values = sess.run(log_sum_exp(
                    tf.constant(a), (0, 2), keepdims))
                self.assertAllClose(test_values, true_values)


class TestLogMeanExp(tf.test.TestCase):
    def test_log_mean_exp(self):
        with self.test_session(use_gpu=True) as sess:
            a = np.array([[[1., 3., 0.2], [0.7, 2., 1e-6]],
                          [[0., 1e6, 1.], [1., 1., 1.]]])
            for keepdims in [True, False]:
                true_values = misc.logsumexp(a, (0, 2), keepdims=keepdims) - \
                              np.log(a.shape[0] * a.shape[2])
                test_values = sess.run(log_mean_exp(
                    tf.constant(a), (0, 2), keepdims))
                self.assertAllClose(test_values, true_values)

            b = np.array([[0., 1e-6, 10.1]])
            test_values = sess.run(log_mean_exp(b, 0, keep_dims=False))
            self.assertTrue(np.abs(test_values - b).max() < 1e-6)


class TestAddNameScope(tf.test.TestCase):
    def test_add_name_scope(self):
        class A:
            @add_name_scope
            def f(self):
                return tf.ones(1)

        a = A()
        node = a.f()
        self.assertEqual(node.name, 'A.f/ones:0')


class TestIfRaise(tf.test.TestCase):
    def test_if_raise(self):
        with self.assertRaisesRegexp(RuntimeError, "if_raise"):
            if_raise(True, RuntimeError("if_raise"))
        if_raise(False, RuntimeError("if_raise"))
