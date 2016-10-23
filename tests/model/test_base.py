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
            a = tf.constant(1., name="as")
            b = tf.exp(a, name="bs")
            c = tf.log(b, name="cs")
            d = tf.neg(b, name="ds")

        b_new = tf.constant(np.e ** 2, name="bs_new")
        d_new = tf.constant(-np.e ** 2, name="ds_new")

        # case 1
        d_out = model.get_output(d)
        assert d_out[0] is d

        # case 2
        c_out, d_out = model.get_output([c, d])
        assert c_out[0] is c
        assert d_out[0] is d

        # case 3
        c_out, d_out = model.get_output([c, d], inputs={b: b_new})
        with tf.Session() as sess:
            c_out_, d_out_ = sess.run([c_out[0], d_out[0]])
            assert np.abs(c_out_ - 2.) < 1e-8
            assert np.abs(d_out_ + np.e ** 2) < 1e-6

        # case 4
        c_out = model.get_output(c, inputs={d: d_new})
        assert c_out[0] is c
        with tf.Session() as sess:
            c_out_ = sess.run(c_out[0])
            assert np.abs(c_out_ - 1.) < 1e-8

        # train_writer = tf.train.SummaryWriter('/tmp/zhusuan',
        #                                       tf.get_default_graph())
        # train_writer.close()

    def test_get_output_merge(self):
        # a -> c -> d
        # b ->/
        with StochasticGraph() as model:
            a = tf.constant(4., name='am')
            b = tf.constant(0., name='bm')
            c = tf.add(a, b, name='cm')
            d = tf.stop_gradient(c, name='dm')

        a_new = tf.constant(10., name='am_new')
        b_new = tf.constant(1., name='bm_new')
        c_new = tf.constant(-1., name='cm_new')

        # case 1
        a_out, b_out, c_out, d_out = model.get_output([a, b, c, d],
                                                      inputs={a: a_new})
        with tf.Session() as sess:
            a_out_, b_out_, c_out_, d_out_ = sess.run([a_out[0], b_out[0],
                                                       c_out[0], d_out[0]])
            assert np.abs(a_out_ - 10.) < 1e-8
            assert np.abs(b_out_ - 0.) < 1e-8
            assert np.abs(c_out_ - 10.) < 1e-8
            assert np.abs(d_out_ - 10.) < 1e-8

        # case 2
        a_out, b_out, c_out, d_out = model.get_output([a, b, c, d],
                                                      inputs={b: b_new})
        with tf.Session() as sess:
            a_out_, b_out_, c_out_, d_out_ = sess.run([a_out[0], b_out[0],
                                                       c_out[0], d_out[0]])
            assert np.abs(a_out_ - 4.) < 1e-8
            assert np.abs(b_out_ - 1.) < 1e-8
            assert np.abs(c_out_ - 5.) < 1e-8
            assert np.abs(d_out_ - 5.) < 1e-8

        # case 3
        a_out, b_out, c_out, d_out = model.get_output([a, b, c, d],
                                                      inputs={c: c_new})
        with tf.Session() as sess:
            a_out_, b_out_, c_out_, d_out_ = sess.run([a_out[0], b_out[0],
                                                       c_out[0], d_out[0]])
            assert np.abs(a_out_ - 4.) < 1e-8
            assert np.abs(b_out_ - 0.) < 1e-8
            assert np.abs(c_out_ - (-1.)) < 1e-8
            assert np.abs(d_out_ - (-1.)) < 1e-8

        # train_writer = tf.train.SummaryWriter('/tmp/zhusuan',
        #                                       tf.get_default_graph())
        # train_writer.close()

    def test_get_output_bridge(self):
        # a -> b -> c -> d -> e
        #       \  ---  /
        with StochasticGraph() as model:
            a = tf.constant([2], dtype=tf.int32, name='ag')
            b = tf.identity(a, name='bg')
            c = tf.neg(b, name='cg')
            d = tf.tile(c, b, name='dg')
            e = tf.square(d, name='eg')

        a_new = tf.constant([3], dtype=tf.int32, name='ag_new')
        b_new = tf.constant([4], dtype=tf.int32, name='bg_new')
        c_new = tf.constant([5], dtype=tf.int32, name='cg_new')
        d_new = tf.constant([5, 5, 5], name='dg_new')

        # case 1
        d_out, e_out = model.get_output([d, e], inputs={a: a_new, c: c_new})
        with tf.Session() as sess:
            d_out_, e_out_ = \
                sess.run([d_out[0], e_out[0]])
            assert (np.abs(d_out_ - np.array([5, 5, 5])).all() < 1e-8)
            assert (np.abs(e_out_ - np.array([25, 25, 25])).all() < 1e-8)

        # case 2
        c_out, e_out = model.get_output([c, e], inputs={a: a_new, b: b_new,
                                                        d: d_new})
        with tf.Session() as sess:
            c_out_, e_out_ = sess.run([c_out[0], e_out[0]])

            assert np.abs(c_out_ - (-4)).all() < 1e-8
            assert (np.abs(e_out_ - np.array([25, 25, 25])).all() < 1e-8)

        # train_writer = tf.train.SummaryWriter('/tmp/zhusuan',
        #                                       tf.get_default_graph())
        # train_writer.close()

    def test_get_output_one_to_many_op(self):
        # tf.unpack
        # a -.---- a0
        #     \ -- a1
        #      \ - a2 -> c
        # b ----------- /
        with StochasticGraph() as model:
            a = tf.zeros([3, 2, 1, 4], name="ao")
            a0, a1, a2 = tf.unpack(a, axis=0)
            b = tf.ones([2, 4, 1], name="bo")
            c = tf.batch_matmul(a2, b, name="co")

        a1_new = tf.ones([2, 1, 4], name="a1_new")
        a_new = tf.ones([3, 2, 1, 4], name="ao_new")
        a2_new = tf.ones([2, 1, 4], name="a2_new") * 2

        # case 1
        a2_out, c_out = model.get_output([a2, c], inputs={a1: a1_new})
        assert a2_out[0] is a2
        assert c_out[0] is c

        # case 2
        a0_out, a2_out, c_out = model.get_output([a0, a2, c],
                                                 inputs={a: a_new, a2: a2_new})
        with tf.Session() as sess:
            a0_out_, a2_out_, c_out_ = sess.run(
                [a0_out[0], a2_out[0], c_out[0]])
            assert np.abs(a0_out_ - np.ones([2, 1, 4])).max() < 1e-8
            assert np.abs(a2_out_ - np.ones([2, 1, 4]) * 2).max() < 1e-8
            assert np.abs(c_out_ - np.array([[8, 8]]).T).max() < 1e-8

        # train_writer = tf.train.SummaryWriter('/tmp/zhusuan',
        #                                       tf.get_default_graph())
        # train_writer.close()

    def test_get_output_many_to_many_op(self):
        # tf.meshgrid
        # / ---- \
        # x - \   \ / -> xx -> w -> v
        # y -> z --.---> zz ------ /
        with StochasticGraph() as model:
            x = tf.constant([1, 2, 3], name='x')
            y = tf.constant([4, 5, 6], name='y')
            z = tf.add(x, y, name='z')
            xx, zz = tf.meshgrid(x, z)
            w = tf.identity(xx, name='w')
            v = tf.add(w, zz, name='v')

        x_new = tf.constant([7, 8, 9], name='x_new')
        y_new = tf.constant([1, -2, -3], name='y_new')
        z_new = tf.constant([-1, -1, -1], name='z_new')
        xx_new = tf.ones([3, 3], dtype=tf.int32, name='xx_new')
        zz_new = tf.zeros([3, 3], dtype=tf.int32, name='zz_new')
        w_new = tf.zeros([3, 3], dtype=tf.int32, name='w_new')

        # case 1
        v_out, w_out = model.get_output([v, w], inputs={x: x_new})
        with tf.Session() as sess:
            v_out_, w_out_ = sess.run([v_out[0], w_out[0]])
            assert np.abs(v_out_ - np.array([[18, 19, 20],
                                             [20, 21, 22],
                                             [22, 23, 24]])).max() < 1e-8
            assert np.abs(w_out_ - np.array([[7, 8, 9],
                                             [7, 8, 9],
                                             [7, 8, 9]])).max() < 1e-8

        # case 2
        v_out = model.get_output(v, inputs={zz: zz_new, y: y_new})
        with tf.Session() as sess:
            v_out_ = sess.run(v_out[0])
            assert np.abs(v_out_ - np.array([[1, 2, 3],
                                             [1, 2, 3],
                                             [1, 2, 3]])).max() < 1e-8

        # case 3
        zz_out, v_out = model.get_output([zz, v],
                                         inputs={x: x_new, z: z_new, w: w_new})
        with tf.Session() as sess:
            zz_out_, v_out_ = sess.run([zz_out[0], v_out[0]])
            assert np.abs(zz_out_ - np.ones((3, 3)) * (-1)).max() < 1e-8
            assert np.abs(v_out_ - np.ones((3, 3)) * (-1)).max() < 1e-8

        # case 4
        zz_out = model.get_output(zz, inputs={xx: xx_new})
        assert zz_out[0] is zz
        with tf.Session() as sess:
            zz_out_ = sess.run(zz_out[0])
            assert np.abs(zz_out_ - np.array([[5, 5, 5],
                                              [7, 7, 7],
                                              [9, 9, 9]])).max() < 1e-8

        # train_writer = tf.train.SummaryWriter('/tmp/zhusuan',
        #                                       tf.get_default_graph())
        # train_writer.close()

    def test_get_output_placeholder_feed(self):
        # a -> c -> c0
        # b - /    /
        #  \ ---- /
        with StochasticGraph() as model:
            a = tf.placeholder(tf.float32, name='ap')
            b = tf.placeholder(tf.int32, name='bp')
            c = tf.expand_dims(a, b, name='cp')
            c0 = tf.split(b, 1, c)[0]

        b_new = tf.placeholder(tf.int32, name='bp_new')
        c0_out = model.get_output(c0, inputs={b: b_new})
        with tf.Session() as sess:
            with pytest.raises(tf.errors.InvalidArgumentError):
                sess.run(c0_out[0], feed_dict={a: np.ones([2, 3]), b: 0})
            c0_out_ = sess.run(c0_out[0], feed_dict={a: np.ones([2, 3]),
                                                     b_new: 0})
            assert np.abs(c0_out_ - np.ones([2, 3])).max() < 1e-8

        # train_writer = tf.train.SummaryWriter('/tmp/zhusuan',
        #                                       tf.get_default_graph())
        # train_writer.close()

    def test_get_output_control_deps(self):
        pass

    def test_get_output_control_flow(self):
        # while_loop, scan, TensorArray
        pass

    def test_get_output_variable(self):
        # w -> y
        # x - /
        with StochasticGraph() as model:
            with tf.variable_scope("weights"):
                w = tf.get_variable("w", shape=[4, 5],
                                    initializer=tf.random_normal_initializer())
                x = tf.ones([2, 5])

    def test_get_output_neural_networks(self):
        pass

    def test_get_output_stochastic_tensor(self):
        pass
