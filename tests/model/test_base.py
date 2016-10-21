#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from itertools import permutations

import pytest
from mock import Mock
import numpy as np

from .context import zhusuan
from zhusuan.model.base import *


class TestStochasticTensor:
    def test_init(self):
        _sample = Mock()

        class _Dist(StochasticTensor):
            def sample(self):
                return _sample
        incomings = [Mock()]
        with StochasticGraph() as model:
            s_tensor = _Dist(incomings)
        assert s_tensor.incomings == incomings
        assert s_tensor.value == _sample

    def test_sample(self):
        mock_graph = StochasticGraph()
        mock_graph.add_stochastic_tensor = Mock(return_value=None)
        with mock_graph:
            s_tensor = StochasticTensor([Mock()])
        with pytest.raises(NotImplementedError):
            s_tensor.sample()

    def test_log_p(self):
        mock_graph = StochasticGraph()
        mock_graph.add_stochastic_tensor = Mock(return_value=None)
        with mock_graph:
            s_tensor = StochasticTensor([Mock()])
        with pytest.raises(NotImplementedError):
            s_tensor.log_p(Mock(), [Mock()])


class TestStochasticGraph:
    def test_init(self):
        with StochasticGraph() as model:
            assert StochasticGraph.get_context() == model
        with pytest.raises(RuntimeError):
            StochasticGraph.get_context()

    def test_add_stochastic_tensor(self):
        s_tensor = Mock(value=Mock())
        model = StochasticGraph()
        model.add_stochastic_tensor(s_tensor)
        assert model.stochastic_tensors[s_tensor.value] == s_tensor

    def _augment_outputs(self, outputs):
        return [[i, None] for i in outputs]

    def test_get_output_inputs_check(self):
        with StochasticGraph() as model:
            a = tf.constant(1.)
        with pytest.raises(TypeError):
            model.get_output(a, inputs=[Mock()])

    def test_get_output_chain(self):
        # a -> b -> c
        with StochasticGraph() as model:
            a = tf.constant(1., name="a")
            b = tf.sqrt(a, name="b")
            c = tf.square(b, name="c")

        for n in range(4):
            for requests in permutations([a, b, c], n):
                assert model.get_output(requests) == \
                    self._augment_outputs(requests)

        a_new = tf.constant(4., name="a_new")
        b_new = tf.constant(2., name="b_new")
        c_new = tf.constant(4., name="c_new")

        # case 1
        c_out = model.get_output(c, inputs={b: b_new})
        with tf.Session() as sess:
            assert np.abs(sess.run(c_out[0]) - 4.) < 1e-8

        # case 2
        b_out, c_out = model.get_output([b, c], inputs={b: b_new})
        with tf.Session() as sess:
            b_out_, c_out_ = sess.run([b_out[0], c_out[0]])
            assert np.abs(b_out_ - 2.) < 1e-8
            assert np.abs(c_out_ - 4.) < 1e-8

        # case 3
        a_out, c_out = model.get_output([a, c], inputs={b: b_new})
        with tf.Session() as sess:
            a_out_, c_out_ = sess.run([a_out[0], c_out[0]])
            assert np.abs(a_out_ - 1.) < 1e-8
            assert np.abs(c_out_ - 4.) < 1e-8

        # case 4
        a_out, b_out, c_out = model.get_output([a, b, c], inputs={a: a_new})
        with tf.Session() as sess:
            a_out_, b_out_, c_out_ = sess.run([a_out[0], b_out[0], c_out[0]])
            assert np.abs(a_out_ - 4.) < 1e-8
            assert np.abs(b_out_ - 2.) < 1e-8
            assert np.abs(c_out_ - 4.) < 1e-8

        # train_writer = tf.train.SummaryWriter('/tmp/zhusuan',
        #                                       tf.get_default_graph())
        # train_writer.close()

    def test_get_output_split(self):
        # a -> b -> c
        #       \-> d
        with StochasticGraph() as model:
            a = tf.constant(1.)
            b = tf.exp(a)
            c = tf.log(b)
            d = tf.neg(c)


    def test_get_output_merge(self):
        pass

    def test_get_output_bridge(self):
        pass

    def test_get_output_control_deps(self):
        pass

    def test_get_output_variable(self):
        pass

    def test_get_output_neural_networks(self):
        pass

    def test_get_output_stochastic_tensor(self):
        pass
