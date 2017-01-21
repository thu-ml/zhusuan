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
from tensorflow.contrib import layers

from .context import zhusuan
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
        assert s_tensor.incomings == incomings
        assert s_tensor.tensor == _sample
        assert s_tensor.s_graph == m1

        _observed = Mock()
        with StochasticGraph(observed={'a': _observed}):
            s_tensor = _Dist('a', incomings)
        assert s_tensor.tensor == _observed

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
        pass


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
            b = a + 1
        with pytest.raises(TypeError):
            model.get_output(a, inputs=[Mock()])

        # Shape mismatch
        with pytest.raises(ValueError):
            model.get_output(a, inputs={a: tf.ones([2])})

        # Tensor -> numpy array
        a_out, _ = model.get_output(a, inputs={a: np.zeros([])})
        assert type(a_out) is tf.Tensor
        with tf.Session() as sess:
            assert np.abs(sess.run(a_out)) < 1e-6

        # Tensor -> Variable
        a_new = tf.Variable(tf.zeros([]))
        b_out, _ = model.get_output(b, inputs={a: a_new})
        assert type(b_out) is tf.Tensor
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            assert np.abs(sess.run(b_out) - 1.) < 1e-6
            sess.run(a_new.assign(-1.))
            assert np.abs(sess.run(b_out)) < 1e-6

    def test_get_output_stochastic_tensor(self):
        # a_mean -- \
        # a_logstd - a -- b_logits - b
        #             \ - c_logits - c
        with StochasticGraph() as model:
            n = tf.placeholder(tf.int32, shape=())
            a_mean = tf.ones([3])
            a_logstd = tf.zeros([3])
            a = Normal(a_mean, a_logstd, sample_dim=0, n_samples=n)
            b_logits = layers.fully_connected(a.value, 5)
            b = Bernoulli(b_logits)
            c_logits = layers.fully_connected(a.value, 4)
            c = Discrete(c_logits)

        # case 1
        a_new = tf.zeros([n, 3])
        b_new = tf.zeros([n, 5])
        a_out, b_out, c_out = model.get_output([a, b, c], inputs={a: a_new})
        with tf.Session() as sess:
            assert a_out[0] is a_new
            assert a_out[1] is not None
            assert c_out[1] is not None
            assert b_out[1] is not None
            sess.run(tf.global_variables_initializer())
            a_out_sample, a_out_logpdf, a_new_ = \
                sess.run([a_out[0], a_out[1], a_new], feed_dict={n: 5})
            assert np.abs(a_out_sample - a_new_).max() < 1e-6

        # case 2
        a_out,  b_out, c_out = model.get_output([a, b, c], inputs={b: b_new})
        with tf.Session() as sess:
            assert b_out[0] is b_new
            assert a_out[0] is a.value
            assert c_out[0] is c.value
            sess.run(tf.global_variables_initializer())
            a_out_, b_out_, c_out_ = sess.run([a_out, b_out, c_out],
                                              feed_dict={n: 1})
            assert a_out_[0].shape == (1, 3)
            assert a_out_[1].shape == (1, 3)
            assert np.abs(b_out_[0]).max() < 1e-6
            assert b_out_[1].shape == (1, 5)
            assert c_out_[0].shape == (1, 4)

        # case 3
        b_out = model.get_output(b)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            b_out_ = sess.run(b_out, feed_dict={n: 2})
            assert b_out_[1].shape == (2, 5)

        # case 4
        n_new = tf.constant(1, tf.int32)
        a_new = tf.zeros([n_new, 3])
        b_out = model.get_output(b, inputs={a: a_new, n: n_new})
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            b_out_ = sess.run(b_out)
            assert b_out_[1].shape == (1, 5)

        # train_writer = tf.train.SummaryWriter('/tmp/zhusuan',
        #                                       tf.get_default_graph())
        # train_writer.close()
