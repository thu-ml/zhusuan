#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from itertools import permutations

import six
import numpy as np
import tensorflow as tf

from zhusuan.model.utils import *
from zhusuan.model.utils import Context
from tests.model._div_op import regular_div, floor_div
from tests.model._true_div_op import true_div


class TestContext(tf.test.TestCase):
    def test_Context(self):
        self.assertEqual(Context.get_contexts(), [])
        with self.assertRaisesRegexp(RuntimeError, "No contexts on the stack"):
            Context.get_context()
        with Context() as context:
            self.assertEqual(Context.get_contexts(), [context])
            self.assertEqual(Context.get_context(), context)
            with Context() as context_inner:
                self.assertEqual(Context.get_contexts(),
                                 [context, context_inner])
                self.assertEqual(Context.get_context(), context_inner)
            self.assertEqual(Context.get_contexts(), [context])
            self.assertEqual(Context.get_context(), context)
        self.assertEqual(Context.get_contexts(), [])
        with self.assertRaisesRegexp(RuntimeError, "No contexts on the stack"):
            Context.get_context()


class TestGetBackwardTensors(tf.test.TestCase):
    def testGetBackwardOpsChain(self):
        # a -> b -> c
        a = tf.placeholder(tf.float32)
        b = tf.sqrt(a)
        c = tf.square(b)
        for n in range(4):
            for seed_tensors in permutations([a, b, c], n):
                if c in seed_tensors:
                    truth = [a.op, b.op, c.op]
                elif b in seed_tensors:
                    truth = [a.op, b.op]
                elif a in seed_tensors:
                    truth = [a.op]
                else:
                    truth = []
                self.assertEqual(get_backward_ops(seed_tensors), truth)

        self.assertEqual(get_backward_ops([c], treat_as_inputs=[b]), [c.op])
        self.assertEqual(
            get_backward_ops([b, c], treat_as_inputs=[b]), [c.op])
        self.assertEqual(
            get_backward_ops([a, c], treat_as_inputs=[b]), [a.op, c.op])

    def testGetBackwardOpsSplit(self):
        # a -> b -> c
        #       \-> d
        a = tf.placeholder(tf.float32)
        b = tf.exp(a)
        c = tf.log(b)
        d = tf.negative(b)
        self.assertEqual(get_backward_ops([d]), [a.op, b.op, d.op])
        self.assertEqual(get_backward_ops([c]), [a.op, b.op, c.op])
        self.assertEqual(
            get_backward_ops([c, d]), [a.op, b.op, c.op, d.op])
        self.assertEqual(get_backward_ops([b, d]), [a.op, b.op, d.op])
        self.assertEqual(get_backward_ops([a, d]), [a.op, b.op, d.op])

        self.assertEqual(
            get_backward_ops([c, d], treat_as_inputs=[b]), [c.op, d.op])
        self.assertEqual(
            get_backward_ops([c], treat_as_inputs=[d]), [a.op, b.op, c.op])

    def testGetBackwardOpsMerge(self):
        # a -> c -> d
        # b ->/
        a = tf.placeholder(tf.float32)
        b = tf.constant(0, dtype=tf.int32)
        c = tf.reduce_sum(a, reduction_indices=b)
        d = tf.stop_gradient(c)
        self.assertEqual(
            get_backward_ops([d]), [a.op, b.op, c.op, d.op])
        self.assertEqual(get_backward_ops([d], treat_as_inputs=[c]), [d.op])
        self.assertEqual(
            get_backward_ops([d], treat_as_inputs=[a]), [b.op, c.op, d.op])

    def testGetBackwardOpsBridge(self):
        # a -> b -> c -> d -> e
        #       \    ---    /
        a = tf.placeholder(tf.int32)
        b = tf.identity(a)
        c = tf.cast(b, tf.float32)
        d = tf.tile(c, b)
        e = tf.tanh(d)
        self.assertEqual(
            get_backward_ops([e]), [a.op, b.op, c.op, d.op, e.op])
        self.assertEqual(get_backward_ops([c]), [a.op, b.op, c.op])
        self.assertEqual(get_backward_ops([e], treat_as_inputs=[c]),
                         [a.op, b.op, d.op, e.op])

    def testGetBackwardOpsControlDeps(self):
        # a -> b - \
        # c -> d - e
        #       \ /
        #        f
        a = tf.placeholder(tf.float32, name='a')
        b = tf.identity(a, name='b')
        c = tf.placeholder(tf.float32, name='c')
        d = tf.identity(c, name='d')
        with tf.control_dependencies([b, d]):
            e = tf.placeholder(tf.float32, name='e')
        with tf.control_dependencies([e, d]):
            f = tf.placeholder(tf.float32, name='f')
        self.assertEqual(get_backward_ops([f]),
                         [a.op, b.op, c.op, d.op, e.op, f.op])
        self.assertEqual(get_backward_ops([d, f]),
                         [c.op, d.op, a.op, b.op, e.op, f.op])

        self.assertEqual(get_backward_ops([f], treat_as_inputs=[b]),
                         [a.op, b.op, c.op, d.op, e.op, f.op])
        self.assertEqual(get_backward_ops([f], treat_as_inputs=[b, c]),
                         [a.op, b.op, d.op, e.op, f.op])
        self.assertEqual(get_backward_ops([f], treat_as_inputs=[d, e]),
                         [a.op, b.op, c.op, d.op, e.op, f.op])
        self.assertEqual(get_backward_ops([d, f], treat_as_inputs=[b]),
                         [c.op, d.op, a.op, b.op, e.op, f.op])

    def test_get_backward_ops_control_flow(self):
        # while_loop, scan, TensorArray
        pass


class _SimpleTensor(TensorArithmeticMixin):

    def __init__(self, value):
        self.value = tf.convert_to_tensor(value)

    @property
    def dtype(self):
        return self.value.dtype


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
        if six.PY2:
            self.assertAlmostEqual(regular_div(3, 2), 1)
            self.assertAlmostEqual(regular_div(3.3, 1.6), 2.0625)
        else:
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

        with tf.Graph().as_default(), self.test_session(use_gpu=True):
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

        with tf.Graph().as_default(), self.test_session(use_gpu=True):
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

        with tf.Graph().as_default(), self.test_session(use_gpu=True):
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
