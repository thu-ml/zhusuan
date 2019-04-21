#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from itertools import permutations

import tensorflow as tf

from zhusuan.framework.utils import *
from zhusuan.framework.utils import Context


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
