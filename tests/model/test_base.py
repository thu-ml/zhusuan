#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from mock import Mock
import tensorflow as tf

from tests.context import zhusuan
from zhusuan.model.base import *
from zhusuan.model.stochastic import *


class TestStochasticTensor(tf.test.TestCase):
    def test_init(self):
        samples = Mock()
        log_probs = Mock()
        probs = Mock()
        sample_func = Mock(return_value=samples)
        log_prob_func = Mock(return_value=log_probs)
        prob_func = Mock(return_value=probs)
        distribution = Mock(sample=sample_func,
                            log_prob=log_prob_func,
                            prob=prob_func,
                            dtype=tf.int32)
        with self.assertRaisesRegexp(
                RuntimeError, "can only be constructed in a BayesianNet"):
            _ = StochasticTensor('test', distribution, 2)
        with BayesianNet() as model:
            s_tensor = StochasticTensor('test', distribution, 2)
        self.assertEqual(s_tensor.name, 'test')
        self.assertTrue(s_tensor.distribution is distribution)
        self.assertEqual(s_tensor.dtype, tf.int32)
        self.assertTrue(s_tensor.net is model)
        self.assertTrue(s_tensor.tensor is samples)
        self.assertTrue(s_tensor.log_prob(None) is log_probs)
        self.assertTrue(s_tensor.prob(None) is probs)

        obs_int32 = tf.placeholder(tf.int32, None)
        obs_float32 = tf.placeholder(tf.float32, None)
        with self.assertRaisesRegexp(
                ValueError,
                "StochasticTensor\(\'a\'\) not compatible.*observed"):
            with BayesianNet(observed={'a': obs_float32}):
                _ = StochasticTensor('a', distribution, 2).tensor
        with BayesianNet(observed={'a': obs_int32}):
            s_tensor = StochasticTensor('a', distribution, 2)
        self.assertTrue(s_tensor.tensor is obs_int32)

    def test_tensor_conversion(self):
        with BayesianNet(observed={'a': 1., 'c': 1.}):
            a = StochasticTensor('a', Mock(dtype=tf.float32), 1)
            b = tf.add(1., a)
            c = StochasticTensor('c', Mock(dtype=tf.float32), 1)
            # tensorflow will try to convert c to the same type with 1 (int32)
            # calling the registered tensor conversion function of c.
            # If failed, it will try not to request the type. So an error
            # will be raised by the operator.
            with self.assertRaisesRegexp(
                    TypeError, "type float32.*not match.*type int32"):
                _ = tf.add(1, c)
        with self.test_session(use_gpu=True):
            self.assertNear(b.eval(), 2., 1e-6)
        with self.assertRaisesRegexp(ValueError, "Ref type not supported"):
            _ = StochasticTensor._to_tensor(a, as_ref=True)

    def test_overload_operator(self):
        with BayesianNet(observed={'a': 1.}):
            a = StochasticTensor('a', Mock(dtype=tf.float32), 1)
            b = a + 1
            # TODO: test all operators
        with self.test_session(use_gpu=True):
            self.assertNear(b.eval(), 2, 1e-6)


class TestBayesianNet(tf.test.TestCase):
    def test_init(self):
        with BayesianNet() as model:
            self.assertTrue(BayesianNet.get_context() is model)
        with self.assertRaisesRegexp(RuntimeError, "No contexts on the stack"):
            BayesianNet.get_context()

    def test_add_stochastic_tensor(self):
        s_tensor = Mock(name=Mock())
        model = BayesianNet()
        model._add_stochastic_tensor(s_tensor)
        self.assertTrue(model._stochastic_tensors[s_tensor.name] is s_tensor)

    def test_query(self):
        # outputs
        a_observed = tf.zeros([])
        with BayesianNet({'a': a_observed}) as model:
            a = Normal('a', 0, 1)
            b = Normal('b', 0, 1)
            c = Normal('c', b, 1)
        self.assertTrue(model.outputs('a') is a_observed)
        b_out, c_out = model.outputs(['b', 'c'])
        self.assertTrue(b_out is b.tensor)
        self.assertTrue(c_out is c.tensor)

        # local_log_prob
        log_pa = model.local_log_prob('a')
        log_pa_t = a.log_prob(a_observed)
        with self.test_session(use_gpu=True) as sess:
            log_pa_out, log_pa_t_out = sess.run([log_pa, log_pa_t])
            self.assertNear(log_pa_out, log_pa_t_out, 1e-6)
        log_pb, log_pc = model.local_log_prob(['b', 'c'])
        log_pb_t, log_pc_t = b.log_prob(b), c.log_prob(c)
        with self.test_session(use_gpu=True) as sess:
            log_pb_out, log_pb_t_out = sess.run([log_pb, log_pb_t])
            log_pc_out, log_pc_t_out = sess.run([log_pc, log_pc_t])
            self.assertNear(log_pb_out, log_pb_t_out, 1e-6)
            self.assertNear(log_pc_out, log_pc_t_out, 1e-6)

        # query
        a_out, log_pa = model.query('a', outputs=True, local_log_prob=True)
        b_outs, c_outs = model.query(['b', 'c'],
                                     outputs=True, local_log_prob=True)
        b_out, log_pb = b_outs
        c_out, log_pc = c_outs
        self.assertTrue(a_out is a.tensor)
        self.assertTrue(b_out is b.tensor)
        self.assertTrue(c_out is c.tensor)
        with self.test_session(use_gpu=True) as sess:
            log_pa_out, log_pa_t_out, log_pb_out, log_pb_t_out, log_pc_out, \
                log_pc_t_out = sess.run([log_pa, log_pa_t, log_pb, log_pb_t,
                                         log_pc, log_pc_t])
            self.assertNear(log_pa_out, log_pa_t_out, 1e-6)
            self.assertNear(log_pb_out, log_pb_t_out, 1e-6)
            self.assertNear(log_pc_out, log_pc_t_out, 1e-6)


class TestReuse(tf.test.TestCase):
    def test_reuse(self):
        @reuse("test")
        def f():
            w = tf.get_variable("w", shape=[])
            return w

        w1 = f()
        w2 = f()
        w3 = f()
        assert w1 is w2
        assert w2 is w3
