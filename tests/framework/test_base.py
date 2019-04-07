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
        static_shape = Mock()
        samples = Mock(shape=static_shape)
        log_probs = Mock()
        probs = Mock()
        sample_func = Mock(return_value=samples)
        log_prob_func = Mock(return_value=log_probs)
        prob_func = Mock(return_value=probs)
        distribution = Mock(sample=sample_func,
                            log_prob=log_prob_func,
                            prob=prob_func,
                            dtype=tf.int32)
        bn = BayesianNet()
        s_tensor = bn.stochastic('test', distribution)
        self.assertEqual(s_tensor.name, 'test')
        self.assertTrue(s_tensor.distribution is distribution)
        self.assertTrue(s_tensor.dist is distribution)
        self.assertEqual(s_tensor.dtype, tf.int32)
        self.assertTrue(s_tensor.tensor is samples)
        self.assertTrue(s_tensor.log_prob(None) is log_probs)
        self.assertTrue(s_tensor.prob(None) is probs)
        self.assertTrue(s_tensor.get_shape() is static_shape)
        self.assertTrue(s_tensor.shape is static_shape)
        self.assertTrue(s_tensor.bn is bn)

        # v3 construction shouldn't fail
        from zhusuan.legacy.framework.stochastic import Normal
        Normal('a', 0., logstd=1.)

        # Test observation checks:
        obs_int32 = tf.placeholder(tf.int32, None)
        obs_float32 = tf.placeholder(tf.float32, None)

        # v3 logic
        with BayesianNet(observed={'a': obs_float32}) as bn:
            s_tensor = Normal('a', 0., logstd=1.)
            self.assertTrue(s_tensor.tensor is obs_float32)
            self.assertTrue(bn['a'].tensor is obs_float32)

        # shape and dtype checks
        with self.assertRaisesRegexp(
                ValueError,
                "Incompatible types of StochasticTensor\(\'a\'\)"):
            bn1 = BayesianNet(observed={'a': obs_float32})
            bn1.stochastic('a', distribution)

        with self.assertRaisesRegexp(
                ValueError,
                "Incompatible shapes of StochasticTensor\(\'a\'\)"):
            bn1 = BayesianNet(observed={'a': tf.zeros((3,), dtype=tf.float32)})
            bn1.normal('a', tf.zeros((2,)), logstd=1.)

    def test_tensor_conversion(self):
        bn = BayesianNet(observed={'a': 1., 'c': 1.})
        a = bn.normal('a', 0., logstd=1.)
        b = tf.add(1., a)
        c = bn.normal('c', 0., logstd=1.)
        # tensorflow will try to convert c to the same type with 1 (int32)
        # calling the registered tensor conversion function of c.
        # If failed, it will try not to request the type. So an error
        # will be raised by the operator.
        with self.assertRaisesRegexp(
                TypeError, "type float32.*not match.*type int32"):
            _ = tf.add(1, c)
        with self.session(use_gpu=True):
            self.assertNear(b.eval(), 2., 1e-6)
        with self.assertRaisesRegexp(ValueError, "Ref type not supported"):
            _ = StochasticTensor._to_tensor(a, as_ref=True)

    def test_overload_operator(self):
        bn = BayesianNet(observed={'a': 1.})
        a = bn.normal('a', 0., logstd=1.)
        b = a + 1
        # TODO: test all operators
        with self.session(use_gpu=True):
            self.assertNear(b.eval(), 2, 1e-6)

    def test_session_run(self):
        with self.session(use_gpu=True) as sess:
            samples = tf.constant([1, 2, 3], dtype=tf.float32)
            # test session.run
            bn = BayesianNet({'t': samples})
            t = bn.normal('t', tf.zeros((3,)), std=1., n_samples=1)
            self.assertAllEqual(sess.run(t), np.asarray([1, 2, 3]))

            # test using as feed dict
            self.assertAllEqual(
                sess.run(tf.identity(t), feed_dict={
                    t: np.asarray([4, 5, 6])
                }),
                np.asarray([4, 5, 6])
            )

    def test_session_run_issue_49(self):
        # test fix for the bug at https://github.com/thu-ml/zhusuan/issues/49
        bn = zs.BayesianNet(observed={})
        x_mean = tf.zeros([1, 2])
        x_logstd = tf.zeros([1, 2])
        x = bn.normal('x', mean=x_mean, logstd=x_logstd, group_ndims=1)

        with self.session(use_gpu=True) as sess:
            sess.run(tf.global_variables_initializer())
            _ = sess.run(x)


class TestBayesianNet(tf.test.TestCase):

    def test_duplicate_nodes(self):
        bn = BayesianNet()
        a = bn.normal('a', 0., logstd=1.)
        with self.assertRaisesRegexp(ValueError, "Names should be unique"):
            a = bn.normal('a', 0., logstd=1.)

    def test_query(self):

        @meta_bayesian_net()
        def build_meta_bn():
            bn = BayesianNet()
            a = bn.normal('a', 0., logstd=1.)
            b = bn.normal('b', 0., logstd=1.)
            c = bn.normal('c', b, logstd=1.)
            return bn, a, b, c

        a_observed = tf.zeros([])
        model, a, b, c = build_meta_bn().observe(a=a_observed)

        # nodes
        self.assertTrue(model.get('b') is b)
        self.assertTrue(model.get(['b', 'c']) == [b, c])

        # outputs
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
        with self.session(use_gpu=True) as sess:
            log_pa_out, log_pa_t_out = sess.run([log_pa, log_pa_t])
            self.assertNear(log_pa_out, log_pa_t_out, 1e-6)
        log_pb, log_pc = model.local_log_prob(['b', 'c'])
        log_pb_t, log_pc_t = b.log_prob(b), c.log_prob(c)
        with self.session(use_gpu=True) as sess:
            log_pb_out, log_pb_t_out = sess.run([log_pb, log_pb_t])
            log_pc_out, log_pc_t_out = sess.run([log_pc, log_pc_t])
            self.assertNear(log_pb_out, log_pb_t_out, 1e-6)
            self.assertNear(log_pc_out, log_pc_t_out, 1e-6)

        # local_log_prob by iterator
        log_pb_2, log_pc_2 = model.local_log_prob(iter(['b', 'c']))
        with self.session(use_gpu=True) as sess:
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
        with self.session(use_gpu=True) as sess:
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
        with self.session(use_gpu=True) as sess:
            log_pb_2_out, log_pb_t_out = sess.run([log_pb_2, log_pb_t])
            log_pc_2_out, log_pc_t_out = sess.run([log_pc_2, log_pc_t])
            self.assertNear(log_pb_2_out, log_pb_t_out, 1e-6)
            self.assertNear(log_pc_2_out, log_pc_t_out, 1e-6)


class TestReuse(tf.test.TestCase):

    def test_legacy_reuse(self):
        @reuse("test")
        def f():
            w = tf.get_variable("w", shape=[])
            return w

        w1 = f()
        w2 = f()
        w3 = f()
        self.assertTrue(w1 is w2)
        self.assertTrue(w2 is w3)

    @staticmethod
    def _generate_bn(var_to_return):
        bn = BayesianNet()
        a_mean = tf.get_variable('a', initializer=tf.constant(0.))
        a = bn.normal('a', a_mean, logstd=1.)
        b = bn.normal('b', 0., logstd=1.)
        c = bn.normal('c', b, logstd=1.)
        return bn, locals()[var_to_return]

    def test_meta_bn(self):
        # the basic usage is tested in TestBayesianNet. corner cases here
        @meta_bayesian_net(scope='scp', reuse_variables=False)
        def build_mbn(var_to_return):
            return TestReuse._generate_bn(var_to_return)

        with tf.variable_scope('you_might_want_do_this'):
            mbn = build_mbn('a_mean')
            _, m1 = mbn.observe()
            with tf.variable_scope('you_might_want_do_this'):
                _, m2 = mbn.observe()
            self.assertNotEqual(m1.name, m2.name)
        with tf.variable_scope('when_you_are_perfectly_conscious'):
            _, m2 = build_mbn('a_mean').observe()
        self.assertNotEquals(m1.name, m2.name)

        @meta_bayesian_net(scope='scp', reuse_variables=True)
        def build_mbn(var_to_return):
            return TestReuse._generate_bn(var_to_return)

        meta_bn = build_mbn('a_mean')
        _, m1 = meta_bn.observe()
        _, m2 = meta_bn.observe()
        _, m3 = build_mbn('a_mean').observe()
        self.assertEquals(m1.name, m2.name)
        self.assertNotEqual(m1.name, m3.name)

        with self.assertRaisesRegexp(ValueError, 'Cannot reuse'):
            @meta_bayesian_net(reuse_variables=True)
            def mbn(var_to_return):
                return TestReuse._generate_bn(var_to_return)
            mbn('a_mean')
