#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np

from zhusuan.distributions.base import *
from tests.distributions import utils
from scipy import stats, misc


from zhusuan.distributions import exponential_family

# TODO: test sample value

class TestExponentialFamily(tf.test.TestCase):
    def test_init(self):
        with self.test_session(use_gpu=True):
            with self.assertRaisesRegexp(
                    ValueError, "Either.*should be passed but not both"):
                Distribution(mean=tf.ones([2, 1]))
            with self.assertRaisesRegexp(
                    ValueError, "Either.*should be passed but not both"):
                Distribution(mean=tf.ones([2, 1]), std=1., logstd=0.)
            with self.assertRaisesRegexp(ValueError,
                                         "should be broadcastable to match"):
                Distribution(mean=tf.ones([2, 1]), logstd=tf.zeros([2, 4, 3]))
            with self.assertRaisesRegexp(ValueError,
                                         "should be broadcastable to match"):
                Distribution(mean=tf.ones([2, 1]), std=tf.ones([2, 4, 3]))

        Distribution(mean=tf.placeholder(tf.float32, [None, 1]),
                     logstd=tf.placeholder(tf.float32, [None, 1, 3]))
        Distribution(mean=tf.placeholder(tf.float32, [None, 1]),
                     std=tf.placeholder(tf.float32, [None, 1, 3]))

    def test_value_shape(self):
        # static
        norm = Distribution(mean=tf.placeholder(tf.float32, None),
                            logstd=tf.placeholder(tf.float32, None))
        self.assertEqual(norm.get_value_shape().as_list(), [])
        norm = Distribution(mean=tf.placeholder(tf.float32, None),
                            std=tf.placeholder(tf.float32, None))
        self.assertEqual(norm.get_value_shape().as_list(), [])

        # dynamic
        self.assertTrue(norm._value_shape().dtype is tf.int32)
        with self.test_session(use_gpu=True):
            self.assertEqual(norm._value_shape().eval().tolist(), [])

        self.assertEqual(norm._value_shape().dtype, tf.int32)

    def test_batch_shape(self):
        utils.test_batch_shape_2parameter_univariate(
            self, Distribution, np.zeros, np.zeros)

    def test_sample_shape(self):
        utils.test_2parameter_sample_shape_same(
            self, Distribution, np.zeros, np.zeros)

    def test_sample_reparameterized(self):
        mean = tf.ones([2, 3])
        logstd = tf.ones([2, 3])
        norm_rep = Distribution(mean, logstd)
        samples = norm_rep.sample(tf.placeholder(tf.int32, shape=[]))
        mean_grads, logstd_grads = tf.gradients(samples, [mean, logstd])
        self.assertTrue(mean_grads is not None)
        self.assertTrue(logstd_grads is not None)

        norm_no_rep = Distribution(mean, logstd, is_reparameterized=False)
        samples = norm_no_rep.sample(tf.placeholder(tf.int32, shape=[]))
        mean_grads, logstd_grads = tf.gradients(samples, [mean, logstd])
        self.assertEqual(mean_grads, None)
        self.assertEqual(logstd_grads, None)

    def test_path_derivative(self):
        mean = tf.ones([2, 3])
        logstd = tf.ones([2, 3])
        n_samples = tf.placeholder(tf.int32, shape=[])

        norm_rep = Distribution(mean, logstd, use_path_derivative=True)
        samples = norm_rep.sample(n_samples)
        log_prob = norm_rep.log_prob(samples)
        mean_path_grads, logstd_path_grads = tf.gradients(log_prob,
                                                          [mean, logstd])
        sample_grads = tf.gradients(log_prob, samples)
        mean_true_grads = tf.gradients(samples, mean, sample_grads)[0]
        logstd_true_grads = tf.gradients(samples, logstd, sample_grads)[0]
        with self.test_session(use_gpu=True) as sess:
            outs = sess.run([mean_path_grads, mean_true_grads,
                             logstd_path_grads, logstd_true_grads],
                            feed_dict={n_samples: 7})
            mean_path, mean_true, logstd_path, logstd_true = outs
            self.assertAllClose(mean_path, mean_true)
            self.assertAllClose(logstd_path, logstd_true)

        norm_no_rep = Distribution(mean, logstd, is_reparameterized=False,
                                   use_path_derivative=True)
        samples = norm_no_rep.sample(n_samples)
        log_prob = norm_no_rep.log_prob(samples)
        mean_path_grads, logstd_path_grads = tf.gradients(log_prob,
                                                          [mean, logstd])
        self.assertTrue(mean_path_grads is None)
        self.assertTrue(mean_path_grads is None)

    def test_log_prob_shape(self):
        utils.test_2parameter_log_prob_shape_same(
            self, Distribution, np.zeros, np.zeros, np.zeros)

    def test_value(self):
        with self.test_session(use_gpu=True):
            def _test_value(given, mean, logstd):
                mean = np.array(mean, np.float32)
                given = np.array(given, np.float32)
                logstd = np.array(logstd, np.float32)
                std = np.exp(logstd)
                target_log_p = stats.norm.logpdf(given, mean, np.exp(logstd))
                target_p = stats.norm.pdf(given, mean, np.exp(logstd))

                norm1 = Distribution(mean, logstd=logstd)
                log_p1 = norm1.log_prob(given)
                self.assertAllClose(log_p1.eval(), target_log_p)
                p1 = norm1.prob(given)
                self.assertAllClose(p1.eval(), target_p)

                norm2 = Distribution(mean, std=std)
                log_p2 = norm2.log_prob(given)
                self.assertAllClose(log_p2.eval(), target_log_p)
                p2 = norm2.prob(given)
                self.assertAllClose(p2.eval(), target_p)

            _test_value(0., 0., 0.)
            _test_value([0.99, 0.9, 9., 99.], 1., [-3., -1., 1., 10.])
            _test_value([7.], [0., 4.], [[1., 2.], [3., 5.]])

    def test_check_numerics(self):
        norm1 = Distribution(tf.ones([1, 2]), logstd=-1e10, check_numerics=True)
        with self.test_session(use_gpu=True):
            with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                         "precision.*Tensor had Inf"):
                norm1.log_prob(0.).eval()

        norm2 = Distribution(tf.ones([1, 2]), logstd=1e3, check_numerics=True)
        with self.test_session(use_gpu=True):
            with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                         "exp\(logstd\).*Tensor had Inf"):
                norm2.sample().eval()

        norm3 = Distribution(tf.ones([1, 2]), std=0., check_numerics=True)
        with self.test_session(use_gpu=True):
            with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                         "log\(std\).*Tensor had Inf"):
                norm3.log_prob(0.).eval()

    def test_dtype(self):
        utils.test_dtype_2parameter(self, Distribution)
