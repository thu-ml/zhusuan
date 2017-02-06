#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from itertools import permutations

import pytest
from mock import Mock
import numpy as np
import tensorflow as tf

from tests.context import zhusuan
from zhusuan.model.base import *
from zhusuan.model.stochastic import *


class TestStochasticTensor:
    def test_init(self):
        _sample = Mock()

        class _Dist(StochasticTensor):
            def sample(self):
                return _sample
        incomings = [Mock()]
        with StochasticGraph() as m1:
            s_tensor = _Dist('a', incomings)
        assert s_tensor.incomings is incomings
        assert s_tensor.tensor is _sample
        assert s_tensor.s_graph is m1

        _observed = tf.ones(2)
        with StochasticGraph(observed={'a': _observed, 'b': _observed}):
            a = _Dist('a', incomings, dtype=tf.float32)
            b = _Dist('b', incomings, dtype=tf.int32)
        assert a.tensor is _observed
        with pytest.raises(ValueError):
            _ = b.tensor

    def test_sample(self):
        mock_graph = StochasticGraph()
        mock_graph.add_stochastic_tensor = Mock(return_value=None)
        with mock_graph:
            s_tensor = StochasticTensor('a', [Mock()])
        with pytest.raises(NotImplementedError):
            s_tensor.sample()

    def test_log_p(self):
        mock_graph = StochasticGraph()
        mock_graph.add_stochastic_tensor = Mock(return_value=None)
        with mock_graph:
            s_tensor = StochasticTensor('a', [Mock()])
        with pytest.raises(NotImplementedError):
            s_tensor.log_prob(Mock())

    def test_tensor_conversion(self):
        with StochasticGraph(observed={'a': 1., 'c': tf.ones([])}) as model:
            a = StochasticTensor('a', [], dtype=tf.float32)
            b = tf.constant(1.) + a
            c = StochasticTensor('c', [], dtype=tf.int32)
            with pytest.raises(ValueError):
                _ = tf.constant(1.) + c
        with tf.Session() as sess:
            assert np.abs(sess.run(b) - 2) < 1e-6
        with pytest.raises(ValueError):
            StochasticTensor._to_tensor(a, as_ref=True)


class TestStochasticGraph:
    def test_init(self):
        with StochasticGraph() as model:
            assert StochasticGraph.get_context() == model
        with pytest.raises(RuntimeError):
            StochasticGraph.get_context()

    def test_add_stochastic_tensor(self):
        s_tensor = Mock(name=Mock())
        model = StochasticGraph()
        model._add_stochastic_tensor(s_tensor)
        assert model.stochastic_tensors[s_tensor.name] == s_tensor

    def test_query(self):
        # outputs
        a_observed = tf.zeros([])
        with StochasticGraph({'a': a_observed}) as model:
            a = Normal('a', 0, 1)
            b = Normal('b', 0, 1)
            c = Normal('c', b, 1)
        assert model.outputs('a') is a_observed
        b_out, c_out = model.outputs(['b', 'c'])
        assert b_out is b.tensor
        assert c_out is c.tensor

        # local_log_prob
        log_pa = model.local_log_prob('a')
        log_pa_t = a.log_prob(a_observed)
        with tf.Session() as sess:
            log_pa_out, log_pa_t_out = sess.run([log_pa, log_pa_t])
            assert np.abs(log_pa_out - log_pa_t_out) < 1e-6
        log_pb, log_pc = model.local_log_prob(['b', 'c'])
        log_pb_t, log_pc_t = b.log_prob(b), c.log_prob(c)
        with tf.Session() as sess:
            log_pb_out, log_pb_t_out = sess.run([log_pb, log_pb_t])
            log_pc_out, log_pc_t_out = sess.run([log_pc, log_pc_t])
            assert np.abs(log_pb_out - log_pb_t_out) < 1e-6
            assert np.abs(log_pc_out - log_pc_t_out) < 1e-6

        # query
        a_out, log_pa = model.query('a', outputs=True, local_log_prob=True)
        b_outs, c_outs = model.query(['b', 'c'],
                                     outputs=True, local_log_prob=True)
        b_out, log_pb = b_outs
        c_out, log_pc = c_outs
        assert a_out is a.tensor
        assert b_out is b.tensor
        assert c_out is c.tensor
        with tf.Session() as sess:
            log_pa_out, log_pa_t_out, log_pb_out, log_pb_t_out, log_pc_out, \
                log_pc_t_out = sess.run([log_pa, log_pa_t, log_pb, log_pb_t,
                                         log_pc, log_pc_t])
            assert np.abs(log_pa_out - log_pa_t_out) < 1e-6
            assert np.abs(log_pb_out - log_pb_t_out) < 1e-6
            assert np.abs(log_pc_out - log_pc_t_out) < 1e-6


def test_reuse():
    @reuse("test")
    def f():
        w = tf.get_variable("w", shape=[])
        return w

    w1 = f()
    w2 = f()
    w3 = f()
    assert w1 is w2
    assert w2 is w3
