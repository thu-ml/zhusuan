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
            s_tensor.log_prob(Mock(), [Mock()])


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
            assert np.abs(c_out_ - 1.) < 1e-6

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
        # a -> b ---> e -----
        # c -> d ----/       \
        #       \ ----------- f
        with StochasticGraph() as model:
            a = tf.placeholder(tf.float32, name='a_deps')
            b = tf.identity(a, name='b_deps')
            c = tf.placeholder(tf.float32, name='c_deps')
            d = tf.identity(c, name='d_deps')
            with tf.control_dependencies([b, d]):
                e = tf.add(1., tf.zeros([2, 2]), name='e_deps')
            with tf.control_dependencies([e, d]):
                f = tf.add(1., tf.ones([2, 2]), name='f_deps')

        d_new = tf.add(1., tf.ones([]), name='d_deps_new')
        e_new = tf.add(1., tf.ones([2, 2]), name='e_deps_new')
        f_out_only_c = model.get_output(f, inputs={d: d_new, e: e_new})
        f_out_only_a = model.get_output(f, inputs={d: d_new})

        with tf.Session() as sess:
            with pytest.raises(tf.errors.InvalidArgumentError):
                sess.run(f)
            with pytest.raises(tf.errors.InvalidArgumentError):
                sess.run(e, feed_dict={a: 1.})
            f_out_only_c_ = sess.run(f_out_only_c[0], feed_dict={c: 1.})
            f_out_only_a_ = sess.run(f_out_only_a[0], feed_dict={a: 1.})
            assert np.abs(f_out_only_c_ - np.ones([2, 2]) - 1.).max() < 1e-8
            assert np.abs(f_out_only_a_ - np.ones([2, 2]) - 1.).max() < 1e-8

        # train_writer = tf.train.SummaryWriter('/tmp/zhusuan',
        #                                       tf.get_default_graph())
        # train_writer.close()

    def test_get_output_control_flow(self):
        # while_loop, scan, TensorArray
        # TODO: add control flow test for StochasticGraph.get_output
        pass

    def test_get_output_variable(self):
        # w -> y
        # x - /
        with StochasticGraph() as model:
            with tf.variable_scope("weights"):
                w = tf.get_variable("w", shape=[4, 5],
                                    initializer=tf.random_normal_initializer())
            x = tf.ones([5, 2], name="x")
            y = tf.matmul(w, x, name="y")

        x_new = tf.zeros([5, 2], name="x_new")
        with tf.variable_scope("weights_new"):
            w_new = tf.get_variable("w_new", shape=[4, 5],
                                    initializer=tf.random_normal_initializer())

        # case 1
        y_out = model.get_output(y, inputs={x: x_new})
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            y_out_ = sess.run(y_out[0])
            assert y_out_.shape == (4, 2)
            assert np.abs(y_out_).max() < 1e-8

        # case 2
        with pytest.raises(TypeError):
            model.get_output(y, inputs={w: w_new})

        # case 3
        with pytest.raises(TypeError):
            model.get_output(y, inputs={x: np.zeros([5, 2])})

        # train_writer = tf.train.SummaryWriter('/tmp/zhusuan',
        #                                       tf.get_default_graph())
        # train_writer.close()

    def test_get_output_fully_connected(self):
        with StochasticGraph() as model:
            x = tf.ones([3, 4], name='x')
            y = layers.fully_connected(x, 10)

        x_new = tf.zeros([3, 4], name='x_new')
        y_out = model.get_output(y, inputs={x: x_new})
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            y_out_ = sess.run(y_out[0])
            assert y_out_.shape == (3, 10)
            assert np.abs(y_out_).max() < 1e-8

        # train_writer = tf.train.SummaryWriter('/tmp/zhusuan',
        #                                       tf.get_default_graph())
        # train_writer.close()

    def test_get_output_convolution(self):
        with StochasticGraph() as model:
            x = tf.ones([2, 5, 5, 3], name='x')
            y = layers.conv2d(x, 2, [3, 3])

        x_new = tf.zeros([2, 5, 5, 3], name='x_new')
        y_out = model.get_output(y, inputs={x: x_new})
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            y_out_ = sess.run(y_out[0])
            assert y_out_.shape == (2, 5, 5, 2)
            assert np.abs(y_out_).max() < 1e-8

        # train_writer = tf.train.SummaryWriter('/tmp/zhusuan',
        #                                       tf.get_default_graph())
        # train_writer.close()

    def test_get_output_batch_norm(self):
        x_value = np.random.random([2, 5, 5, 3])
        w_value = np.random.random([3, 3, 3, 2])
        is_training_t = tf.placeholder(tf.bool, name='is_training_t')
        x_t = tf.constant(x_value, dtype=tf.float32, name='x_t')
        y_t = layers.conv2d(x_t, 2, [3, 3], normalizer_fn=layers.batch_norm,
                            normalizer_params={'is_training': is_training_t,
                                               'updates_collections': None},
                            weights_initializer=tf.constant_initializer(
                                w_value))
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            y_test_1 = sess.run(y_t, feed_dict={is_training_t: False})
            sess.run(y_t, feed_dict={is_training_t: True})
            y_test_2 = sess.run(y_t, feed_dict={is_training_t: False})

        with StochasticGraph() as model:
            is_training = tf.placeholder(tf.bool, name='is_training')
            x = tf.constant(x_value, dtype=tf.float32, name='x')
            y = layers.conv2d(x, 2, [3, 3], normalizer_fn=layers.batch_norm,
                              normalizer_params={'is_training': is_training,
                                                 'updates_collections': None},
                              weights_initializer=tf.constant_initializer(
                                  w_value))
        x_new = tf.constant(x_value, dtype=tf.float32, name='x')
        y_out = model.get_output(y, inputs={x: x_new})
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            y_out_1 = sess.run(y_out[0], feed_dict={is_training: False})
            y_out_2 = sess.run(y_out[0], feed_dict={is_training: False})
            sess.run(y_out[0], feed_dict={is_training: True})
            y_out_3 = sess.run(y_out[0], feed_dict={is_training: False})
            assert np.abs(y_out_1 - y_out_2).max() < 1e-6
            assert np.abs(y_out_1 - y_out_3).max() > 1e-6

        assert np.abs(y_test_1 - y_out_1).max() < 1e-6
        assert np.abs(y_test_2 - y_out_3).max() < 1e-6

        # TODO: deal with name_scope conflicts when copying batch_norm
        # train_writer = tf.train.SummaryWriter('/tmp/zhusuan',
        #                                       tf.get_default_graph())
        # train_writer.close()

    def test_get_output_stochastic_tensor(self):
        # a_mean -- \
        # a_logstd - a -- b_logits - b
        #             \ - c_logits - c
        with StochasticGraph() as model:
            n = tf.placeholder(tf.int32, shape=())
            # n = 5
            a_mean = tf.ones([3])
            a_logstd = tf.zeros([3])
            a = Normal(a_mean, a_logstd, sample_dim=0, n_samples=n)
            b_logits = layers.fully_connected(a.value, 5)
            b = Bernoulli(b_logits)
            c_logits = layers.fully_connected(a.value, 4)
            c = Discrete(c_logits)

        a_new = tf.zeros([n, 3])
        b_new = tf.zeros([n, 5])
        a_out, b_out, c_out = model.get_output([a, b, c], inputs={a: a_new})
        with tf.Session() as sess:
            assert a_out[0] is a_new
            # assert a_out[1] is not None
            assert c_out[1] is not None
            assert b_out[1] is not None
            sess.run(tf.initialize_all_variables())
            a_out_sample, a_out_logpdf, a_new_ = \
                sess.run([a_out[0], a_out[1], a_new], feed_dict={n: 5})
            assert np.abs(a_out_sample - a_new_).max() < 1e-6
        a_out,  b_out, c_out = model.get_output([a, b, c], inputs={b: b_new})
        with tf.Session() as sess:
            assert b_out[0] is b_new
            assert a_out[0] is a.value
            assert c_out[0] is c.value
            sess.run(tf.initialize_all_variables())
            a_out_, b_out_, c_out_ = sess.run([a_out, b_out, c_out],
                                              feed_dict={n: 1})
            assert a_out_[0].shape == (1, 3)
            assert a_out_[1].shape == (1, 3)
            assert np.abs(b_out_[0] - np.zeros([1, 5])).max() < 1e-6
            assert b_out_[1].shape == (1, 5)
            assert c_out_[0].shape == (1, 4)

        b_out = model.get_output(b)
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            b_out_ = sess.run(b_out, feed_dict={n: 2})
            assert b_out_[1].shape == (2, 5)
        n_new = tf.constant(1, tf.int32)
        b_out = model.get_output(b, inputs={a: a_new, n: n_new})
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            b_out_ = sess.run(b_out)
            assert b_out_[1].shape == (1, 5)
