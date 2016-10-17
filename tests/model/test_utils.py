#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
