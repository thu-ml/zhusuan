#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from itertools import permutations

import pytest

from .context import zhusuan
from zhusuan.model.utils import *


def test_Context():
    assert Context.get_contexts() == []
    with pytest.raises(RuntimeError):
        Context.get_context()
    with Context() as context:
        assert Context.get_contexts() == [context]
        assert Context.get_context() == context
        with Context() as context_inner:
            assert Context.get_contexts() == [context, context_inner]
            assert Context.get_context() == context_inner
        assert Context.get_contexts() == [context]
        assert Context.get_context() == context
    assert Context.get_contexts() == []
    with pytest.raises(RuntimeError):
        Context.get_context()


def test_get_unique_graph():
    g1 = tf.Graph()
    with g1.as_default():
        a = tf.constant(1.)
        b = a + 1
    assert get_unique_graph([a, b]) is g1
    g2 = tf.Graph()
    with g2.as_default():
        c = tf.constant(1.)
    assert get_unique_graph(c) is g2
    with pytest.raises(ValueError):
        get_unique_graph([a, c])
    with pytest.raises(TypeError):
        get_unique_graph([a, 1.])


def test_get_backward_ops_chain():
    # a -> b -> c
    a = tf.placeholder(tf.float32)
    # tf.sqrt() is a unary operator
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
            assert get_backward_ops(seed_tensors) == truth

    assert get_backward_ops(c, treat_as_inputs=[b]) == [c.op]
    assert get_backward_ops([b, c], treat_as_inputs=[b]) == [c.op]
    assert get_backward_ops([a, c], treat_as_inputs=[b]) == [a.op, c.op]


def test_get_backward_ops_split():
    # a -> b -> c
    #       \-> d
    a = tf.placeholder(tf.float32)
    b = tf.exp(a)
    c = tf.log(b)
    d = tf.neg(b)
    assert get_backward_ops(d) == [a.op, b.op, d.op]
    assert get_backward_ops(c) == [a.op, b.op, c.op]
    assert get_backward_ops([c, d]) == [a.op, b.op, c.op, d.op]
    assert get_backward_ops([b, d]) == [a.op, b.op, d.op]
    assert get_backward_ops([a, d]) == [a.op, b.op, d.op]

    assert get_backward_ops([c, d], treat_as_inputs=[b]) == [c.op, d.op]
    assert get_backward_ops(c, treat_as_inputs=[d]) == [a.op, b.op, c.op]


def test_get_backward_ops_merge():
    # a -> c -> d
    # b ->/
    a = tf.placeholder(tf.float32)
    b = tf.constant(0, dtype=tf.int32)
    c = tf.reduce_sum(a, reduction_indices=b)
    d = tf.stop_gradient(c)
    assert get_backward_ops(d) == [a.op, b.op, c.op, d.op]
    assert get_backward_ops(d, treat_as_inputs=[c]) == [d.op]
    assert get_backward_ops(d, treat_as_inputs=[a]) == [b.op, c.op, d.op]


def test_get_backward_ops_bridge():
    # a -> b -> c -> d -> e
    #       \  ---  /
    a = tf.placeholder(tf.int32)
    b = tf.identity(a)
    c = tf.cast(b, tf.float32)
    d = tf.tile(c, b)
    e = tf.tanh(d)
    assert get_backward_ops(e) == [a.op, b.op, c.op, d.op, e.op]
    assert get_backward_ops(c) == [a.op, b.op, c.op]
    assert get_backward_ops(e, treat_as_inputs=[c]) == [a.op, b.op, d.op, e.op]


def test_get_backward_tensors_control_deps():
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
    assert get_backward_ops(f) == [a.op, b.op, c.op, d.op, e.op, f.op]
    assert get_backward_ops([d, f]) == [c.op, d.op, a.op, b.op, e.op, f.op]

    assert get_backward_ops(f, treat_as_inputs=[b]) == [
        a.op, b.op, c.op, d.op, e.op, f.op]
    assert get_backward_ops(f, treat_as_inputs=[b, c]) == [
        a.op, b.op, d.op, e.op, f.op]
    assert get_backward_ops(f, treat_as_inputs=[d, e]) == [
        a.op, b.op, c.op, d.op, e.op, f.op]
    assert get_backward_ops([d, f], treat_as_inputs=[b]) == [
        c.op, d.op, a.op, b.op, e.op, f.op]


def test_get_backward_ops_control_flow():
    # while_loop, scan, TensorArray
    pass
