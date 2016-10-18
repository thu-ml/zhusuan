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


def test_get_backward_tensors_chain():
    # a -> b -> c
    a = tf.placeholder(tf.float32)
    # tf.sqrt() is a unary operator
    b = tf.sqrt(a)
    c = tf.square(b)
    for n in range(4):
        for seed_tensors in permutations([a, b, c], n):
            if c in seed_tensors:
                truth = [a, b, c]
            elif b in seed_tensors:
                truth = [a, b]
            elif a in seed_tensors:
                truth = [a]
            else:
                truth = []
            assert get_backward_tensors(seed_tensors) == truth

    assert get_backward_tensors(c, treat_as_inputs=[b]) == [b, c]
    assert get_backward_tensors([b, c], treat_as_inputs=[b]) == [b, c]
    assert get_backward_tensors([a, c], treat_as_inputs=[b]) == [a, b, c]


def test_get_backward_tensors_split():
    # a -> b -> c
    #       \-> d
    a = tf.placeholder(tf.float32)
    b = tf.exp(a)
    c = tf.log(b)
    d = tf.neg(b)
    assert get_backward_tensors(d) == [a, b, d]
    assert get_backward_tensors(c) == [a, b, c]
    assert get_backward_tensors([c, d]) == [a, b, c, d]
    assert get_backward_tensors([b, d]) == [a, b, d]
    assert get_backward_tensors([a, d]) == [a, b, d]

    assert get_backward_tensors([c, d], treat_as_inputs=[b]) == [b, c, d]
    assert get_backward_tensors(c, treat_as_inputs=[d]) == [a, b, c]


def test_get_backward_tensors_merge():
    # a -> c -> d
    # b ->/
    a = tf.placeholder(tf.float32)
    b = tf.constant(0, dtype=tf.int32)
    c = tf.reduce_sum(a, reduction_indices=b)
    d = tf.stop_gradient(c)
    assert get_backward_tensors(d) == [a, b, c, d]
    assert get_backward_tensors(d, treat_as_inputs=[c]) == [c, d]


def test_get_backward_tensors_bridge():
    # a -> b -> c -> d -> e
    #       \  ---  /
    a = tf.placeholder(tf.int32)
    b = tf.identity(a)
    c = tf.cast(b, tf.float32)
    d = tf.tile(c, b)
    e = tf.tanh(d)
    assert get_backward_tensors(e) == [a, b, c, d, e]
    assert get_backward_tensors(c) == [a, b, c]
    assert get_backward_tensors(e, treat_as_inputs=[c]) == [c, a, b, d, e]


def test_get_backward_tensors_control_deps():
    a = tf.placeholder(tf.float32)
    b = tf.identity(a)
    c = tf.placeholder(tf.float32)
    d = tf.identity(c)
    with tf.control_dependencies([b, d]):
        e = tf.placeholder(tf.float32)
    with tf.control_dependencies([e, d]):
        f = tf.placeholder(tf.float32)
    assert get_backward_tensors(f) == [a, b, c, d, e, f]
    assert get_backward_tensors([d, f]) == [c, d, a, b, e, f]

    assert get_backward_tensors(f, treat_as_inputs=[b]) == [b, c, d, e, f]
    assert get_backward_tensors(f, treat_as_inputs=[d, e]) == [e, d, f]
    assert get_backward_tensors([d, f], treat_as_inputs=[b]) == [c, d, b, e, f]
