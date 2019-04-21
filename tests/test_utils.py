#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
from scipy.special import logsumexp
import numpy as np
import six

from zhusuan.utils import *
from zhusuan.utils import add_name_scope, if_raise, log_sum_exp
from tests._div_op import regular_div, floor_div
from tests._true_div_op import true_div


class _SimpleTensor(TensorArithmeticMixin):

    def __init__(self, value):
        self.value = tf.convert_to_tensor(value)

    @property
    def dtype(self):
        return self.value.dtype

    @property
    def shape(self):
        return self.value.shape


def _to_tensor(value, dtype=None, name=None, as_ref=False):
    if dtype and not dtype.is_compatible_with(value.dtype):
        raise ValueError('Incompatible type conversion requested to type '
                         '%s for variable of type %s' %
                         (dtype.name, value.dtype.name))
    if as_ref:
        raise ValueError('%r: Ref type not supported.' % value)
    return value.value


tf.register_tensor_conversion_function(_SimpleTensor, _to_tensor)


class ArithMixinTestCase(tf.test.TestCase):

    def test_prerequisite(self):
        # Tensorflow has deprecated Python 2 division semantics,
        # regular division in Python 3 is true division.
        # if six.PY2:
        #     self.assertAlmostEqual(regular_div(3, 2), 1)
        #     self.assertAlmostEqual(regular_div(3.3, 1.6), 2.0625)
        # else:
        self.assertAlmostEqual(regular_div(3, 2), 1.5)
        self.assertAlmostEqual(regular_div(3.3, 1.6), 2.0625)
        self.assertAlmostEqual(true_div(3, 2), 1.5)
        self.assertAlmostEqual(true_div(3.3, 1.6), 2.0625)
        self.assertAlmostEqual(floor_div(3, 2), 1)
        self.assertAlmostEqual(floor_div(3.3, 1.6), 2.0)

    def test_unary_op(self):
        def check_op(name, func, x):
            x_tensor = tf.convert_to_tensor(x)
            ans = func(x_tensor)
            res = tf.convert_to_tensor(func(_SimpleTensor(x_tensor)))
            self.assertEqual(
                res.dtype, ans.dtype,
                msg='Result dtype does not match answer after unary operator '
                    '%s is applied: %r vs %r (x is %r).' %
                    (name, res.dtype, ans.dtype, x)
            )
            res_val = res.eval()
            ans_val = ans.eval()
            np.testing.assert_equal(
                res_val, ans_val,
                err_msg='Result value does not match answer after unary '
                        'operator %s is applied: %r vs %r (x is %r).' %
                        (name, res_val, ans_val, x)
            )

        with tf.Graph().as_default(), self.session(use_gpu=True):
            int_data = np.asarray([1, -2, 3], dtype=np.int32)
            float_data = np.asarray([1.1, -2.2, 3.3], dtype=np.float32)
            bool_data = np.asarray([True, False, True], dtype=np.bool)

            check_op('abs', abs, int_data)
            check_op('abs', abs, float_data)
            check_op('neg', (lambda v: -v), int_data)
            check_op('neg', (lambda v: -v), float_data)
            check_op('invert', (lambda v: ~v), bool_data)

    def test_binary_op(self):
        def check_op(name, func, x, y):
            x_tensor = tf.convert_to_tensor(x)
            y_tensor = tf.convert_to_tensor(y)
            ans = func(x_tensor, y_tensor)
            res_1 = tf.convert_to_tensor(
                func(_SimpleTensor(x_tensor), y_tensor))
            res_2 = tf.convert_to_tensor(
                func(x_tensor, _SimpleTensor(y_tensor)))
            res_3 = tf.convert_to_tensor(
                func(_SimpleTensor(x_tensor), _SimpleTensor(y_tensor)))

            for tag, res in [('left', res_1), ('right', res_2),
                             ('both', res_3)]:
                self.assertEqual(
                    res.dtype, ans.dtype,
                    msg='Result dtype does not match answer after %s binary '
                        'operator %s is applied: %r vs %r (x is %r, y is %r).'
                        % (tag, name, res.dtype, ans.dtype, x, y)
                )
                res_val = res.eval()
                ans_val = ans.eval()
                np.testing.assert_equal(
                    res_val, ans_val,
                    err_msg='Result value does not match answer after %s '
                            'binary operator %s is applied: %r vs %r '
                            '(x is %r, y is %r).' %
                            (tag, name, res_val, ans_val, x, y)
                )

        def run_ops(x, y, ops):
            for name, func in six.iteritems(ops):
                check_op(name, func, x, y)

        arith_ops = {
            'add': lambda x, y: x + y,
            'sub': lambda x, y: x - y,
            'mul': lambda x, y: x * y,
            'div': regular_div,
            'truediv': true_div,
            'floordiv': floor_div,
            'mod': lambda x, y: x % y,
        }

        logical_ops = {
            'and': lambda x, y: x & y,
            'or': lambda x, y: x | y,
            'xor': lambda x, y: x ^ y,
        }

        relation_ops = {
            'lt': lambda x, y: x < y,
            'le': lambda x, y: x <= y,
            'gt': lambda x, y: x > y,
            'ge': lambda x, y: x >= y,
        }

        with tf.Graph().as_default(), self.session(use_gpu=True):
            # arithmetic operators
            run_ops(np.asarray([-4, 5, 6], dtype=np.int32),
                    np.asarray([1, -2, 3], dtype=np.int32),
                    arith_ops)
            run_ops(np.asarray([-4.4, 5.5, 6.6], dtype=np.float32),
                    np.asarray([1.1, -2.2, 3.3], dtype=np.float32),
                    arith_ops)

            # it seems that tf.pow(x, y) does not support negative integers
            # yet, so we individually test this operator here.
            check_op('pow',
                     (lambda x, y: x ** y),
                     np.asarray([-4, 5, 6], dtype=np.int32),
                     np.asarray([1, 2, 3], dtype=np.int32))
            check_op('pow',
                     (lambda x, y: x ** y),
                     np.asarray([-4.4, 5.5, 6.6], dtype=np.float32),
                     np.asarray([1.1, -2.2, 3.3], dtype=np.float32))

            # logical operators
            run_ops(np.asarray([True, False, True, False], dtype=np.bool),
                    np.asarray([True, True, False, False], dtype=np.bool),
                    logical_ops)

            # relation operators
            run_ops(np.asarray([1, -2, 3, -4, 5, 6, -4, 5, 6], dtype=np.int32),
                    np.asarray([1, -2, 3, 1, -2, 3, -4, 5, 6], dtype=np.int32),
                    relation_ops)
            run_ops(
                np.asarray([1.1, -2.2, 3.3, -4.4, 5.5, 6.6, -4.4, 5.5, 6.6],
                           dtype=np.float32),
                np.asarray([1.1, -2.2, 3.3, 1.1, -2.2, 3.3, -4.4, 5.5, 6.6],
                           dtype=np.float32),
                relation_ops
            )

    def test_getitem(self):
        def check_getitem(x, y, xx, yy):
            ans = tf.convert_to_tensor(x[y])
            res = xx[yy]

            self.assertEqual(
                res.dtype, ans.dtype,
                msg='Result dtype does not match answer after getitem '
                    'is applied: %r vs %r (x is %r, y is %r, xx is %r, '
                    'yy is %r).' % (res.dtype, ans.dtype, x, y, xx, yy)
            )
            res_val = res.eval()
            ans_val = ans.eval()
            np.testing.assert_equal(
                res_val, ans_val,
                err_msg='Result value does not match answer after '
                        'getitem is applied: %r vs %r (x is %r, y is %r, '
                        'xx is %r, yy is %r).' %
                        (res_val, ans_val, x, y, xx, yy)
            )

        class _SliceGenerator(object):
            def __getitem__(self, item):
                return item
        sg = _SliceGenerator()

        with tf.Graph().as_default(), self.session(use_gpu=True):
            data = np.asarray([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int32)
            indices_or_slices = [
                0,
                -1,
                # TensorFlow has not supported array index yet.
                # np.asarray([0, 3, 2, 6], dtype=np.int32),
                # np.asarray([-1, -2, -3], dtype=np.int32),
                sg[0:],
                sg[:1],
                sg[:: 2],
                sg[-1:],
                sg[: -1],
                sg[:: -1],
            ]
            for s in indices_or_slices:
                x_tensor = tf.convert_to_tensor(data)
                x_simple_tensor = _SimpleTensor(x_tensor)
                check_getitem(data, s, x_simple_tensor, s)

                if not isinstance(s, slice):
                    y_tensor = tf.convert_to_tensor(s)
                    y_simple_tensor = _SimpleTensor(y_tensor)
                    check_getitem(data, s, x_simple_tensor, y_tensor)
                    check_getitem(data, s, x_simple_tensor, y_simple_tensor)
                    check_getitem(data, s, x_tensor, y_simple_tensor)

    def test_disallowed_operator(self):
        with tf.Graph().as_default():
            with self.assertRaisesRegexp(
                    TypeError, '_SimpleTensor object is not iterable'):
                _ = iter(_SimpleTensor(1))

            with self.assertRaisesRegexp(
                    TypeError, 'Using a `_SimpleTensor` object as a Python '
                               '`bool` is not allowed'):
                _ = not _SimpleTensor(1)

            with self.assertRaisesRegexp(
                    TypeError, 'Using a `_SimpleTensor` object as a Python '
                               '`bool` is not allowed'):
                if _SimpleTensor(1):
                    pass


class TestLogSumExp(tf.test.TestCase):
    def test_log_sum_exp(self):
        with self.session(use_gpu=True) as sess:
            a = np.array([[[1., 3., 0.2], [0.7, 2., 1e-6]],
                          [[0., 1e6, 1.], [1., 1., 1.]]])
            for keepdims in [True, False]:
                true_values = logsumexp(a, (0, 2), keepdims=keepdims)
                test_values = sess.run(log_sum_exp(
                    tf.constant(a), (0, 2), keepdims))
                self.assertAllClose(test_values, true_values)
                self.assertEqual(test_values.dtype, true_values.dtype)


class TestLogMeanExp(tf.test.TestCase):
    def test_log_mean_exp(self):
        with self.session(use_gpu=True) as sess:
            a = np.array([[[1., 3., 0.2], [0.7, 2., 1e-6]],
                          [[0., 1e6, 1.], [1., 1., 1.]]])
            for keepdims in [True, False]:
                true_values = logsumexp(a, (0, 2), keepdims=keepdims) - \
                              np.log(a.shape[0] * a.shape[2])
                test_values = sess.run(log_mean_exp(
                    tf.constant(a), (0, 2), keepdims))
                self.assertAllClose(test_values, true_values)
                self.assertEqual(test_values.dtype, true_values.dtype)

            b = np.array([[0., 1e-6, 10.1]])
            test_values = sess.run(log_mean_exp(b, 0, keepdims=False))
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
