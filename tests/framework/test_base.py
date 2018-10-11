#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from mock import Mock
import numpy as np
import tensorflow as tf

import zhusuan as zs
from zhusuan.framework import *


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
        with BayesianNet() as bn:
            s_tensor = bn.stochastic('test', distribution, n_samples=2)
        self.assertEqual(s_tensor.name, 'test')
        self.assertTrue(s_tensor.distribution is distribution)
        self.assertEqual(s_tensor.dtype, tf.int32)
        self.assertTrue(s_tensor.bn is bn)
        self.assertTrue(s_tensor.tensor is samples)
        self.assertTrue(s_tensor.log_prob(None) is log_probs)
        self.assertTrue(s_tensor.prob(None) is probs)

        obs_int32 = tf.placeholder(tf.int32, None)
        obs_float32 = tf.placeholder(tf.float32, None)
        with self.assertRaisesRegexp(
                ValueError,
                "Incompatible types of StochasticTensor\(\'a\'\)"):
            bn = BayesianNet(observed={'a': obs_float32})
            _ = bn.stochastic('a', distribution, n_samples=2).tensor
        bn2 = BayesianNet(observed={'a': obs_int32})
        s_tensor = bn2.stochastic('a', distribution, n_samples=2)
        self.assertTrue(s_tensor.tensor is obs_int32)

    def test_tensor_conversion(self):
        with BayesianNet(observed={'a': 1., 'c': 1.}) as bn:
            a = bn.normal('a', 0., std=1.)
            b = tf.add(1., a)
            c = bn.normal('c', 0., std=1.)
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
        with BayesianNet(observed={'a': 1.}) as bn:
            a = bn.normal('a', 0., std=1.)
            b = a + 1
            # TODO: test all operators
        with self.test_session(use_gpu=True):
            self.assertNear(b.eval(), 2, 1e-6)

    def test_session_run(self):
        with self.test_session(use_gpu=True) as sess:
            samples = tf.constant([1., 2., 3.])
            # test session.run
            bn = BayesianNet(observed={'t': samples})
            t = bn.normal('t', tf.zeros((3,)), std=tf.zeros((3,)))
            self.assertAllEqual(sess.run(t), np.asarray([1., 2., 3.]))

            # test using as feed dict
            self.assertAllEqual(
                sess.run(tf.identity(t), feed_dict={
                    t: np.asarray([4, 5, 6])
                }),
                np.asarray([4, 5, 6])
            )

    def test_session_run_issue_49(self):
        # test fix for the bug at https://github.com/thu-ml/zhusuan/issues/49
        with zs.BayesianNet(observed={}) as model:
            x_mean = tf.zeros([1, 2])
            x_logstd = tf.zeros([1, 2])
            x = zs.Normal('x', mean=x_mean, logstd=x_logstd, group_ndims=1)

        with self.test_session(use_gpu=True) as sess:
            sess.run(tf.global_variables_initializer())
            _ = sess.run(x)


class TestBayesianNet(tf.test.TestCase):
    def test_init(self):
        with BayesianNet() as model:
            self.assertTrue(BayesianNet.get_context() is model)
        with self.assertRaisesRegexp(RuntimeError, "No contexts on the stack"):
            BayesianNet.get_context()

    def test_query(self):
        # outputs
        a_observed = tf.zeros([])
        with BayesianNet({'a': a_observed}) as model:
            a = model.normal('a', 0., logstd=1.)
            b = model.normal('b', 0., logstd=1.)
            c = model.normal('c', b, logstd=1.)
        self.assertTrue(model.outputs('a') is a_observed)
        b_out, c_out = model.outputs(['b', 'c'])
        self.assertTrue(b_out is b.tensor)
        self.assertTrue(c_out is c.tensor)

        # outputs by iterator
        b_out_2, c_out_2 = model.outputs(iter(['b', 'c']))
        self.assertIs(b_out_2, b_out)
        self.assertIs(c_out_2, c_out)

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

        # local_log_prob by iterator
        log_pb_2, log_pc_2 = model.local_log_prob(iter(['b', 'c']))
        with self.test_session(use_gpu=True) as sess:
            log_pb_2_out, log_pb_t_out = sess.run([log_pb_2, log_pb_t])
            log_pc_2_out, log_pc_t_out = sess.run([log_pc_2, log_pc_t])
            self.assertNear(log_pb_2_out, log_pb_t_out, 1e-6)
            self.assertNear(log_pc_2_out, log_pc_t_out, 1e-6)

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

        # query by iterator
        (b_out_2, log_pb_2), (c_out_2, log_pc_2) = \
            model.query(iter(['b', 'c']), outputs=True, local_log_prob=True)
        self.assertIs(b_out_2, b_out)
        self.assertIs(c_out_2, c_out)
        with self.test_session(use_gpu=True) as sess:
            log_pb_2_out, log_pb_t_out = sess.run([log_pb_2, log_pb_t])
            log_pc_2_out, log_pc_t_out = sess.run([log_pc_2, log_pc_t])
            self.assertNear(log_pb_2_out, log_pb_t_out, 1e-6)
            self.assertNear(log_pc_2_out, log_pc_t_out, 1e-6)


class TestReuse(tf.test.TestCase):
    def test_reuse(self):
        @reuse("test")
        def f():
            w = tf.get_variable("w", shape=[])
            return w

        w1 = f()
        w2 = f()
        w3 = f()
        self.assertTrue(w1 is w2)
        self.assertTrue(w2 is w3)
