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
            a = tf.constant(1., name="a")
            b = tf.exp(a, name="b")
            c = tf.log(b, name="c")
            d = tf.neg(b, name="d")

        b_new = tf.constant(np.e ** 2, name="b_new")
        d_new = tf.constant(-np.e ** 2, name="d_new")

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
        # tf.split
        pass

    def test_get_output_many_to_many_op(self):
        with StochasticGraph() as model:
            x = tf.constant([1, 2, 3], name='x')
            y = tf.constant([4, 5, 6], name='y')
            z = tf.add(x, y, name='z')
            mat_x, mat_z = tf.meshgrid(x, z)
            mat_xx = tf.identity(mat_x, name='mat_xx')
            mat = tf.add(mat_xx, mat_z, name='mat')

        for n in range(7):
            for requests in permutations([x, y, z, mat_x, mat_xx, mat], n):
                assert model.get_output(requests) == \
                    self._augment_outputs(requests)

        x_new = tf.constant([7, 8, 9], name='x_new')
        y_new = tf.constant([1, -2, -3], name='y_new')
        z_new = tf.constant([-1, -1, -1], name='z_new')
        mat_x_new = tf.ones([3, 3], dtype=tf.int32, name='mat_x_new')
        mat_z_new = tf.zeros([3, 3], dtype=tf.int32, name='mat_z_new')
        mat_xx_new = tf.zeros([3, 3], dtype=tf.int32, name='mat_xx_new')

        # case 1
        mat_out, mat_xx_out = model.get_output([mat, mat_xx], inputs={x: x_new})
        with tf.Session() as sess:
            mat_out_, mat_xx_out_ = sess.run([mat_out[0], mat_xx_out[0]])
            assert np.abs(mat_out_ - np.array([[18, 20, 22],
                                              [19, 21, 23],
                                              [20, 22, 24]])).all() < 1e-8
            assert np.abs(mat_xx_out_ - np.array([[7, 7, 7],
                                                 [8, 8, 8],
                                                 [9, 9, 9]])).all() < 1e-8
        # case 2
        mat_out = model.get_output(mat, inputs={mat_z: mat_z_new, y: y_new})
        with tf.Session() as sess:
            mat_out_ = sess.run(mat_out[0])
            assert np.abs(mat_out_ - np.array([[3, 1, 1],
                                              [4, 2, 2],
                                              [5, 3, 3]])).all() < 1e-8
        # case 3
        mat_z_out, mat_out = model.get_output([mat_z, mat],
                                              inputs={x: x_new, z: z_new,
                                                      mat_xx: mat_xx_new})
        with tf.Session() as sess:
            mat_z_out_, mat_out_ = sess.run([mat_z_out[0], mat_out[0]])
            assert np.abs(mat_z_out_ - np.array([[-1, -1, -1],
                                                [-1, -1, -1],
                                                [-1, -1, -1]])).all() < 1e-8
            assert np.abs(mat_out_ - np.array([[-1, -1, -1],
                                               [-1, -1, -1],
                                               [-1, -1, -1]])).all() < 1e-8
        # case 4
        mat_xx_out, mat_out = model.get_output([mat_xx, mat],
                                               inputs={mat_x: mat_x_new})
        with tf.Session() as sess:
            mat_xx_out_, mat_out_ = sess.run([mat_xx_out[0], mat_out[0]])
            assert np.abs(mat_xx_out_ - np.array([[1, 1, 1],
                                                 [1, 1, 1],
                                                 [1, 1, 1]])).all() < 1e-8
            assert np.abs(mat_out_ - np.array([[6, 8, 10],
                                               [6, 8, 10],
                                               [6, 8, 10]])).all() < 1e-8

        # train_writer = tf.train.SummaryWriter('/tmp/zhusuan',
        #                                       tf.get_default_graph())
        # train_writer.close()

    def test_get_output_control_deps(self):
        pass

    def test_get_output_variable(self):
        pass

    def test_get_output_neural_networks(self):
        pass

    def test_get_output_stochastic_tensor(self):
        pass
