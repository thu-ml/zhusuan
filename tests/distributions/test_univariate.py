#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
from scipy import stats, misc

from tests.distributions import utils
from zhusuan.distributions.univariate import *


# TODO: test sample value

class TestNormal(tf.test.TestCase):
    def test_init(self):
        with self.test_session(use_gpu=True):
            with self.assertRaisesRegexp(
                    ValueError, "Either.*should be passed but not both"):
                Normal(mean=tf.ones([2, 1]))
            with self.assertRaisesRegexp(
                    ValueError, "Either.*should be passed but not both"):
                Normal(mean=tf.ones([2, 1]), std=1., logstd=0.)
            with self.assertRaisesRegexp(ValueError,
                                         "should be broadcastable to match"):
                Normal(mean=tf.ones([2, 1]), logstd=tf.zeros([2, 4, 3]))
            with self.assertRaisesRegexp(ValueError,
                                         "should be broadcastable to match"):
                Normal(mean=tf.ones([2, 1]), std=tf.ones([2, 4, 3]))

        Normal(mean=tf.placeholder(tf.float32, [None, 1]),
               logstd=tf.placeholder(tf.float32, [None, 1, 3]))
        Normal(mean=tf.placeholder(tf.float32, [None, 1]),
               std=tf.placeholder(tf.float32, [None, 1, 3]))

    def test_value_shape(self):
        # static
        norm = Normal(mean=tf.placeholder(tf.float32, None),
                      logstd=tf.placeholder(tf.float32, None))
        self.assertEqual(norm.get_value_shape().as_list(), [])
        norm = Normal(mean=tf.placeholder(tf.float32, None),
                      std=tf.placeholder(tf.float32, None))
        self.assertEqual(norm.get_value_shape().as_list(), [])

        # dynamic
        self.assertTrue(norm._value_shape().dtype is tf.int32)
        with self.test_session(use_gpu=True):
            self.assertEqual(norm._value_shape().eval().tolist(), [])

        self.assertEqual(norm._value_shape().dtype, tf.int32)

    def test_batch_shape(self):
        utils.test_batch_shape_2parameter_univariate(
            self, Normal, np.zeros, np.zeros)

    def test_sample_shape(self):
        utils.test_sample_shape_2parameter_univariate(
            self, Normal, np.zeros, np.zeros)

    def test_sample_reparameterized(self):
        mean = tf.ones([2, 3])
        logstd = tf.ones([2, 3])
        norm_rep = Normal(mean, logstd)
        samples = norm_rep.sample(tf.placeholder(tf.int32, shape=[]))
        mean_grads, logstd_grads = tf.gradients(samples, [mean, logstd])
        self.assertTrue(mean_grads is not None)
        self.assertTrue(logstd_grads is not None)

        norm_no_rep = Normal(mean, logstd, is_reparameterized=False)
        samples = norm_no_rep.sample(tf.placeholder(tf.int32, shape=[]))
        mean_grads, logstd_grads = tf.gradients(samples, [mean, logstd])
        self.assertEqual(mean_grads, None)
        self.assertEqual(logstd_grads, None)

    def test_log_prob_shape(self):
        utils.test_log_prob_shape_2parameter_univariate(
            self, Normal, np.zeros, np.zeros, np.zeros)

    def test_value(self):
        with self.test_session(use_gpu=True):
            def _test_value(given, mean, logstd):
                mean = np.array(mean, np.float32)
                given = np.array(given, np.float32)
                logstd = np.array(logstd, np.float32)
                std = np.exp(logstd)
                target_log_p = stats.norm.logpdf(given, mean, np.exp(logstd))
                target_p = stats.norm.pdf(given, mean, np.exp(logstd))

                norm1 = Normal(mean, logstd=logstd)
                log_p1 = norm1.log_prob(given)
                self.assertAllClose(log_p1.eval(), target_log_p)
                p1 = norm1.prob(given)
                self.assertAllClose(p1.eval(), target_p)

                norm2 = Normal(mean, std=std)
                log_p2 = norm2.log_prob(given)
                self.assertAllClose(log_p2.eval(), target_log_p)
                p2 = norm2.prob(given)
                self.assertAllClose(p2.eval(), target_p)

            _test_value(0., 0., 0.)
            _test_value([0.99, 0.9, 9., 99.], 1., [-10., -1., 1., 10.])
            _test_value([7.], [0., 4.], [[1., 2.], [3., 5.]])

    def test_check_numerics(self):
        norm1 = Normal(tf.ones([1, 2]), logstd=-1e10, check_numerics=True)
        with self.test_session(use_gpu=True):
            with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                         "precision.*Tensor had Inf"):
                norm1.log_prob(0.).eval()

        norm2 = Normal(tf.ones([1, 2]), logstd=1e3, check_numerics=True)
        with self.test_session(use_gpu=True):
            with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                         "exp\(logstd\).*Tensor had Inf"):
                norm2.sample().eval()

        norm3 = Normal(tf.ones([1, 2]), std=0., check_numerics=True)
        with self.test_session(use_gpu=True):
            with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                         "log\(std\).*Tensor had Inf"):
                norm3.log_prob(0.).eval()

    def test_dtype(self):
        utils.test_dtype_2parameter(self, Normal)


class TestBernoulli(tf.test.TestCase):
    def test_value_shape(self):
        # static
        bernoulli = Bernoulli(tf.placeholder(tf.float32, None))
        self.assertEqual(bernoulli.get_value_shape().as_list(), [])

        # dynamic
        self.assertTrue(bernoulli._value_shape().dtype is tf.int32)
        with self.test_session(use_gpu=True):
            self.assertEqual(bernoulli._value_shape().eval().tolist(), [])

        self.assertEqual(bernoulli._value_shape().dtype, tf.int32)

    def test_batch_shape(self):
        utils.test_batch_shape_1parameter(
            self, Bernoulli, np.zeros, is_univariate=True)

    def test_sample_shape(self):
        utils.test_sample_shape_1parameter_univariate(
            self, Bernoulli, np.zeros)

    def test_log_prob_shape(self):
        utils.test_log_prob_shape_1parameter_univariate(
            self, Bernoulli, np.zeros, np.zeros)

    def test_value(self):
        with self.test_session(use_gpu=True):
            def _test_value(logits, given):
                logits = np.array(logits, np.float32)
                given = np.array(given, np.float32)
                bernoulli = Bernoulli(logits)
                log_p = bernoulli.log_prob(given)
                target_log_p = stats.bernoulli.logpmf(
                    given, 1. / (1. + np.exp(-logits)))
                self.assertAllClose(log_p.eval(), target_log_p)
                p = bernoulli.prob(given)
                target_p = stats.bernoulli.pmf(
                    given, 1. / (1. + np.exp(-logits)))
                self.assertAllClose(p.eval(), target_p)

            _test_value(0., [0, 1])
            _test_value([-50., -10., -50.], [1, 1, 0])
            _test_value([0., 4.], [[0, 1], [0, 1]])
            _test_value([[2., 3., 1.], [5., 7., 4.]],
                        np.ones([3, 1, 2, 3], dtype=np.int32))

    def test_dtype(self):
        utils.test_dtype_1parameter_discrete(self, Bernoulli)


class TestCategorical(tf.test.TestCase):
    def test_init_check_shape(self):
        with self.test_session(use_gpu=True):
            with self.assertRaisesRegexp(ValueError, "should have rank"):
                Categorical(logits=tf.zeros([]))

    def test_init_n_categories(self):
        cat = Categorical(tf.ones([10]))
        self.assertTrue(isinstance(cat.n_categories, int))
        self.assertEqual(cat.n_categories, 10)
        cat2 = Categorical(tf.placeholder(tf.float32, [3, None]))
        self.assertTrue(cat2.n_categories is not None)

        with self.test_session(use_gpu=True):
            logits = tf.placeholder(tf.float32, None)
            cat3 = Categorical(logits)
            self.assertEqual(
                cat3.n_categories.eval(feed_dict={logits: np.ones([10])}), 10)
            with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                         "should have rank"):
                cat3.n_categories.eval(feed_dict={logits: 1.})

    def test_value_shape(self):
        # static
        cat = Categorical(tf.placeholder(tf.float32, None))
        self.assertEqual(cat.get_value_shape().as_list(), [])

        # dynamic
        self.assertTrue(cat._value_shape().dtype is tf.int32)
        with self.test_session(use_gpu=True):
            self.assertEqual(cat._value_shape().eval().tolist(), [])

        self.assertEqual(cat._value_shape().dtype, tf.int32)

    def test_batch_shape(self):
        # static
        def _test_static(logits_shape):
            logits = tf.placeholder(tf.float32, logits_shape)
            cat = Categorical(logits)
            if cat.get_batch_shape():
                self.assertEqual(cat.get_batch_shape().as_list(),
                                 logits_shape[:-1])
            else:
                self.assertEqual(None, logits_shape)

        _test_static([2])
        _test_static([2, 3])
        _test_static([2, 1, 4])
        _test_static([None])
        _test_static([None, 3, 5])
        _test_static([1, None, 3])
        _test_static(None)

        # dynamic
        with self.test_session(use_gpu=True):
            def _test_dynamic(logits_shape):
                logits = tf.placeholder(tf.float32, None)
                cat = Categorical(logits)
                self.assertTrue(cat.batch_shape.dtype is tf.int32)
                self.assertEqual(
                    cat.batch_shape.eval(
                        feed_dict={logits: np.zeros(logits_shape)}).tolist(),
                    logits_shape[:-1])

            _test_dynamic([2])
            _test_dynamic([2, 3])
            _test_dynamic([2, 1, 4])

    def test_sample_shape(self):
        def _test_static(logits_shape, n_samples, target_shape):
            logits = tf.placeholder(tf.float32, logits_shape)
            cat = Categorical(logits)
            samples = cat.sample(n_samples)
            if samples.get_shape():
                self.assertEqual(samples.get_shape().as_list(), target_shape)
            else:
                self.assertEqual(None, target_shape)

        _test_static([2], None, [])
        _test_static([2], 1, [1])
        _test_static([2, 3], None, [2])
        _test_static([2, 3], 1, [1, 2])
        _test_static([5], 2, [2])
        _test_static([1, 2, 4], None, [1, 2])
        _test_static([1, 2, 4], 1, [1, 1, 2])
        _test_static([None, 2], tf.placeholder(tf.int32, []), [None, None])
        _test_static([None, 1, 10], None, [None, 1])
        _test_static(None, None, None)
        _test_static(None, 1, None)
        _test_static([3, None], 2, [2, 3])

        with self.test_session(use_gpu=True):
            def _test_dynamic(logits_shape, n_samples, target_shape):
                logits = tf.placeholder(tf.float32, None)
                cat = Categorical(logits)
                samples = cat.sample(n_samples)
                self.assertEqual(
                    tf.shape(samples).eval(
                        feed_dict={logits: np.zeros(logits_shape)}).tolist(),
                    target_shape)

            _test_dynamic([2], 1, [1])
            _test_dynamic([2, 3], 1, [1, 2])
            _test_dynamic([1, 3], 2, [2, 1])
            _test_dynamic([2, 1, 5], 3, [3, 2, 1])

    def test_log_prob_shape(self):
        def _test_static(logits_shape, given_shape, target_shape):
            logits = tf.placeholder(tf.float32, logits_shape)
            given = tf.placeholder(tf.int32, given_shape)
            cat = Categorical(logits)
            log_p = cat.log_prob(given)
            if log_p.get_shape():
                self.assertEqual(log_p.get_shape().as_list(), target_shape)
            else:
                self.assertEqual(None, target_shape)

        _test_static([2, 3], [2], [2])
        _test_static([5], [], [])
        _test_static([1, 2, 4], [1], [1, 2])
        _test_static([3, 1, 5], [1, 4], [3, 4])
        _test_static([None, 2, 4], [3, None], [3, 2])
        _test_static([None, 2], [None, 1, 1], [None, 1, None])
        _test_static(None, [2, 2], None)
        _test_static([3, None], [3, 2, 1, 1], [3, 2, 1, 3])
        with self.assertRaisesRegexp(ValueError, "broadcast to match"):
            _test_static([2, 3, 5], [1, 2], None)

        with self.test_session(use_gpu=True):
            def _test_dynamic(logits_shape, given_shape, target_shape):
                logits = tf.placeholder(tf.float32, None)
                cat = Categorical(logits)
                given = tf.placeholder(tf.int32, None)
                log_p = cat.log_prob(given)
                self.assertEqual(
                    tf.shape(log_p).eval(
                        feed_dict={logits: np.zeros(logits_shape),
                                   given: np.zeros(given_shape,
                                                   np.int32)}).tolist(),
                    target_shape)

            _test_dynamic([2, 3, 3], [1, 3], [2, 3])
            _test_dynamic([1, 3, 4], [2, 2, 3], [2, 2, 3])
            _test_dynamic([1, 5, 1], [1, 2, 3, 1], [1, 2, 3, 5])
            with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                         "Incompatible shapes"):
                _test_dynamic([2, 3, 5], [1, 2], None)

    def test_value(self):
        with self.test_session(use_gpu=True):
            def _test_value(logits, given):
                logits = np.array(logits, np.float32)
                normalized_logits = logits - misc.logsumexp(
                    logits, axis=-1, keepdims=True)
                given = np.array(given, np.int32)
                cat = Categorical(logits)
                log_p = cat.log_prob(given)

                def _one_hot(x, depth):
                    n_elements = x.size
                    ret = np.zeros((n_elements, depth))
                    ret[np.arange(n_elements), x.flat] = 1
                    return ret.reshape(list(x.shape) + [depth])

                target_log_p = np.sum(_one_hot(
                    given, logits.shape[-1]) * normalized_logits, -1)
                self.assertAllClose(log_p.eval(), target_log_p)
                p = cat.prob(given)
                target_p = np.sum(_one_hot(
                    given, logits.shape[-1]) * np.exp(normalized_logits), -1)
                self.assertAllClose(p.eval(), target_p)

            _test_value([0.], [0, 0, 0])
            _test_value([-50., -10., -50.], [0, 1, 2, 1])
            _test_value([0., 4.], [[0, 1], [0, 1]])
            _test_value([[2., 3., 1.], [5., 7., 4.]],
                        np.ones([3, 1, 1], dtype=np.int32))

    def test_dtype(self):
        utils.test_dtype_1parameter_discrete(self, Categorical)


class TestUniform(tf.test.TestCase):
    def test_init_check_shape(self):
        with self.test_session(use_gpu=True):
            with self.assertRaisesRegexp(ValueError,
                                         "should be broadcastable to match"):
                Uniform(minval=tf.zeros([2, 1]), maxval=tf.ones([2, 4, 3]))

        Uniform(tf.placeholder(tf.float32, [None, 1]),
                tf.placeholder(tf.float32, [None, 1, 3]))

    def test_value_shape(self):
        # static
        unif = Uniform(minval=tf.placeholder(tf.float32, None),
                       maxval=tf.placeholder(tf.float32, None))
        self.assertEqual(unif.get_value_shape().as_list(), [])

        # dynamic
        self.assertTrue(unif._value_shape().dtype is tf.int32)
        with self.test_session(use_gpu=True):
            self.assertEqual(unif._value_shape().eval().tolist(), [])

        self.assertEqual(unif._value_shape().dtype, tf.int32)

    def test_batch_shape(self):
        utils.test_batch_shape_2parameter_univariate(
            self, Uniform, np.zeros, np.ones)

    def test_sample_shape(self):
        utils.test_sample_shape_2parameter_univariate(
            self, Uniform, np.zeros, np.ones)

    def test_sample_reparameterized(self):
        minval = tf.ones([2, 3])
        maxval = tf.ones([2, 3])
        unif_rep = Uniform(minval, maxval)
        samples = unif_rep.sample(tf.placeholder(tf.int32, shape=[]))
        minval_grads, maxval_grads = tf.gradients(samples, [minval, maxval])
        self.assertTrue(minval_grads is not None)
        self.assertTrue(maxval_grads is not None)

        unif_no_rep = Uniform(minval, maxval, is_reparameterized=False)
        samples = unif_no_rep.sample(tf.placeholder(tf.int32, shape=[]))
        minval_grads, maxval_grads = tf.gradients(samples, [minval, maxval])
        self.assertEqual(minval_grads, None)
        self.assertEqual(maxval_grads, None)

    def test_log_prob_shape(self):
        utils.test_log_prob_shape_2parameter_univariate(
            self, Uniform, np.zeros, np.ones, np.zeros)

    def test_value(self):
        with self.test_session(use_gpu=True):
            def _test_value(minval, maxval, given):
                minval = np.array(minval, np.float32)
                maxval = np.array(maxval, np.float32)
                given = np.array(given, np.float32)
                unif = Uniform(minval, maxval)
                log_p = unif.log_prob(given)
                target_log_p = stats.uniform.logpdf(given, minval,
                                                    maxval - minval)
                self.assertAllClose(log_p.eval(), target_log_p)
                p = unif.prob(given)
                target_p = stats.uniform.pdf(given, minval, maxval - minval)
                self.assertAllClose(p.eval(), target_p)

            # Uniform semantics different from scipy at maxval.
            self.assertEqual(Uniform(0., 1.).log_prob(1).eval(), -np.inf)
            _test_value(0., 1., [-1., 0., 0.5, 2.])
            _test_value([-1e10, -1], [1, 1e10], 0.)
            _test_value([0., -1.], [[[1., 2.], [3., 5.], [4., 9.]]], [7.])

    def test_check_numerics(self):
        unif = Uniform(0., [0., 1.], check_numerics=True)
        with self.test_session(use_gpu=True):
            with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                         "p.*Tensor had Inf"):
                unif.log_prob(0.).eval()

    def test_dtype(self):
        utils.test_dtype_2parameter(self, Uniform)


class TestGamma(tf.test.TestCase):
    def test_init_check_shape(self):
        with self.test_session(use_gpu=True):
            with self.assertRaisesRegexp(ValueError,
                                         "should be broadcastable to match"):
                Gamma(alpha=tf.ones([2, 1]), beta=tf.ones([2, 4, 3]))

        Gamma(tf.placeholder(tf.float32, [None, 1]),
              tf.placeholder(tf.float32, [None, 1, 3]))

    def test_value_shape(self):
        # static
        gamma = Gamma(alpha=tf.placeholder(tf.float32, None),
                      beta=tf.placeholder(tf.float32, None))
        self.assertEqual(gamma.get_value_shape().as_list(), [])

        # dynamic
        self.assertTrue(gamma._value_shape().dtype is tf.int32)
        with self.test_session(use_gpu=True):
            self.assertEqual(gamma._value_shape().eval().tolist(), [])
        self.assertEqual(gamma._value_shape().dtype, tf.int32)

    def test_batch_shape(self):
        utils.test_batch_shape_2parameter_univariate(
            self, Gamma, np.ones, np.ones)

    def test_sample_shape(self):
        utils.test_sample_shape_2parameter_univariate(
            self, Gamma, np.ones, np.ones)

    def test_log_prob_shape(self):
        utils.test_log_prob_shape_2parameter_univariate(
            self, Gamma, np.ones, np.ones, np.ones)

    def test_value(self):
        with self.test_session(use_gpu=True):
            def _test_value(alpha, beta, given):
                alpha = np.array(alpha, np.float32)
                beta = np.array(beta, np.float32)
                given = np.array(given, np.float32)
                gamma = Gamma(alpha, beta)
                log_p = gamma.log_prob(given)
                target_log_p = stats.gamma.logpdf(given, alpha,
                                                  scale=1. / beta)
                self.assertAllClose(log_p.eval(), target_log_p)
                p = gamma.prob(given)
                target_p = stats.gamma.pdf(given, alpha, scale=1. / beta)
                self.assertAllClose(p.eval(), target_p)

            _test_value(1., 1., [1., 10., 1e8])
            _test_value([0.5, 1., 2., 3., 5., 7.5, 9.],
                        [2., 2., 2., 1., 0.5, 1., 1.],
                        np.transpose([np.arange(1, 20)]))
            _test_value([1e-8, 1e8], [[1., 1e8], [1e-8, 5.]], [7.])

    def test_check_numerics(self):
        alpha = tf.placeholder(tf.float32, [])
        beta = tf.placeholder(tf.float32, [])
        given = tf.placeholder(tf.float32, [])
        gamma = Gamma(alpha, beta, check_numerics=True)
        log_p = gamma.log_prob(given)
        with self.test_session(use_gpu=True):
            with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                         "log\(given\).*Tensor had Inf"):
                log_p.eval(feed_dict={alpha: 1., beta: 1., given: 0.})
            with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                         "log\(beta\).*Tensor had NaN"):
                log_p.eval(feed_dict={alpha: 1., beta: -1., given: 1.})
            with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                         "log\(alpha\).*Tensor had NaN"):
                log_p.eval(feed_dict={alpha: -1.5, beta: 1., given: 1.})

    def test_dtype(self):
        utils.test_dtype_2parameter(self, Gamma)


class TestBeta(tf.test.TestCase):
    def test_init_check_shape(self):
        with self.test_session(use_gpu=True):
            with self.assertRaisesRegexp(ValueError,
                                         "should be broadcastable to match"):
                Beta(alpha=tf.ones([2, 1]), beta=tf.ones([2, 4, 3]))

        Beta(tf.placeholder(tf.float32, [None, 1]),
             tf.placeholder(tf.float32, [None, 1, 3]))

    def test_value_shape(self):
        # static
        dist = Beta(alpha=tf.placeholder(tf.float32, None),
                    beta=tf.placeholder(tf.float32, None))
        self.assertEqual(dist.get_value_shape().as_list(), [])

        # dynamic
        self.assertTrue(dist._value_shape().dtype is tf.int32)
        with self.test_session(use_gpu=True):
            self.assertEqual(dist._value_shape().eval().tolist(), [])
        self.assertEqual(dist._value_shape().dtype, tf.int32)

    def test_batch_shape(self):
        utils.test_batch_shape_2parameter_univariate(
            self, Beta, np.ones, np.ones)

    def test_sample_shape(self):
        utils.test_sample_shape_2parameter_univariate(
            self, Beta, np.ones, np.ones)

    def test_log_prob_shape(self):
        utils.test_log_prob_shape_2parameter_univariate(
            self, Beta, np.ones, np.ones, np.ones)

    def test_value(self):
        with self.test_session(use_gpu=True):
            def _test_value(alpha, beta, given):
                alpha = np.array(alpha, np.float32)
                beta = np.array(beta, np.float32)
                given = np.array(given, np.float32)
                dist = Beta(alpha, beta)
                log_p = dist.log_prob(given)
                target_log_p = stats.beta.logpdf(given, alpha, beta)
                self.assertAllClose(log_p.eval(), target_log_p)
                p = dist.prob(given)
                target_p = stats.beta.pdf(given, alpha, beta)
                self.assertAllClose(p.eval(), target_p)

            # _test_value(1., 1., [0., 0.5, 1.])
            _test_value([0.5, 5., 1., 2., 2.],
                        [0.5, 1., 3., 2., 5.],
                        np.transpose([np.arange(0.1, 1, 0.1)]))
            _test_value([[1e-8], [1e8]], [[1., 1e8], [1e-8, 1.]], [0.7])

    def test_check_numerics(self):
        alpha = tf.placeholder(tf.float32, [])
        beta = tf.placeholder(tf.float32, [])
        given = tf.placeholder(tf.float32, [])
        dist = Beta(alpha, beta, check_numerics=True)
        log_p = dist.log_prob(given)
        with self.test_session(use_gpu=True):
            with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                         "log\(given\).*Tensor had Inf"):
                log_p.eval(feed_dict={alpha: 1., beta: 1., given: 0.})
            with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                         "log\(1 - given\).*Tensor had Inf"):
                log_p.eval(feed_dict={alpha: 1., beta: 1., given: 1.})
            with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                         "lgamma\(beta\).*Tensor had Inf"):
                log_p.eval(feed_dict={alpha: 2., beta: -1., given: 0.5})
            with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                         "lgamma\(alpha\).*Tensor had Inf"):
                log_p.eval(feed_dict={alpha: 0., beta: 1., given: 0.5})

    def test_dtype(self):
        utils.test_dtype_2parameter(self, Beta)


class TestPoisson(tf.test.TestCase):
    def test_value_shape(self):
        # static
        poisson = Poisson(tf.placeholder(tf.float32, None))
        self.assertEqual(poisson.get_value_shape().as_list(), [])

        # dynamic
        self.assertTrue(poisson._value_shape().dtype is tf.int32)
        with self.test_session(use_gpu=True):
            self.assertEqual(poisson._value_shape().eval().tolist(), [])
        self.assertEqual(poisson._value_shape().dtype, tf.int32)

    def test_batch_shape(self):
        utils.test_batch_shape_1parameter(
            self, Poisson, np.ones, is_univariate=True)

    def test_sample_shape(self):
        utils.test_sample_shape_1parameter_univariate(
            self, Poisson, np.ones)

    def test_log_prob_shape(self):
        utils.test_log_prob_shape_1parameter_univariate(
            self, Poisson, np.ones, np.ones)

    def test_value(self):
        with self.test_session(use_gpu=True):
            def _test_value(rate, given):
                rate = np.array(rate, np.float32)
                given = np.array(given, np.float32)
                poisson = Poisson(rate)
                log_p = poisson.log_prob(given)
                target_log_p = stats.poisson.logpmf(given, rate)
                self.assertAllClose(log_p.eval(), target_log_p)
                p = poisson.prob(given)
                target_p = stats.poisson.pmf(given, rate)
                self.assertAllClose(p.eval(), target_p)

            _test_value(1, [0, 1, 2, 3, 4, 5, 6])
            _test_value([5, 1, 5], [0, 0, 1])
            _test_value([10000, 1], [[100, 0], [0, 100]])
            _test_value([[1, 10, 100], [999, 99, 9]],
                        np.ones([3, 1, 2, 3], dtype=np.int32))
            _test_value([[1, 10, 100], [999, 99, 9]],
                        100 * np.ones([3, 1, 2, 3], dtype=np.int32))

    def test_check_numerics(self):
        rate = tf.placeholder(tf.float32, [])
        given = tf.placeholder(tf.int32, [])
        poisson = Poisson(rate, check_numerics=True)
        log_p = poisson.log_prob(given)
        with self.test_session(use_gpu=True):
            with self.assertRaisesRegexp(
                    tf.errors.InvalidArgumentError,
                    "lgamma\(given \+ 1\).*Tensor had Inf"):
                log_p.eval(feed_dict={rate: 1., given: -2})
            with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                         "log\(rate\).*Tensor had NaN"):
                log_p.eval(feed_dict={rate: -1., given: 1})

    def test_dtype(self):
        utils.test_dtype_1parameter_discrete(self, Poisson)


class TestBinomial(tf.test.TestCase):
    def test_init_n(self):
        dist = Binomial(tf.ones([2]), 10)
        self.assertTrue(isinstance(dist.n_experiments, int))
        self.assertEqual(dist.n_experiments, 10)
        with self.assertRaisesRegexp(ValueError, "must be positive"):
            _ = Binomial(tf.ones([2]), 0)

        with self.test_session(use_gpu=True):
            logits = tf.placeholder(tf.float32, None)
            n_experiments = tf.placeholder(tf.int32, None)
            dist2 = Binomial(logits, n_experiments)
            with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                         "should be a scalar"):
                dist2.n_experiments.eval(feed_dict={logits: [1.],
                                                    n_experiments: [10]})
            with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                         "must be positive"):
                dist2.n_experiments.eval(feed_dict={logits: [1.],
                                                    n_experiments: 0})

    def test_value_shape(self):
        # static
        binomial = Binomial(tf.placeholder(tf.float32, None), 10)
        self.assertEqual(binomial.get_value_shape().as_list(), [])

        # dynamic
        self.assertTrue(binomial._value_shape().dtype is tf.int32)
        with self.test_session(use_gpu=True):
            self.assertEqual(binomial._value_shape().eval().tolist(), [])
        self.assertEqual(binomial._value_shape().dtype, tf.int32)

    def test_batch_shape(self):
        def _distribution(param):
            return Binomial(param, 10)
        utils.test_batch_shape_1parameter(
            self, _distribution, np.ones, is_univariate=True)

    def test_sample_shape(self):
        def _distribution(param):
            return Binomial(param, 10)
        utils.test_sample_shape_1parameter_univariate(
            self, _distribution, np.ones)

    def test_log_prob_shape(self):
        def _distribution(param):
            return Binomial(param, 10)
        utils.test_log_prob_shape_1parameter_univariate(
            self, _distribution, np.ones, np.ones)

    def test_value(self):
        with self.test_session(use_gpu=True):
            def _test_value(logits, n_experiments, given):
                logits = np.array(logits, np.float64)
                given = np.array(given, np.float64)
                binomial = Binomial(logits, n_experiments)
                log_p = binomial.log_prob(given)
                target_log_p = stats.binom.logpmf(
                    given, n_experiments, 1 / (1. + np.exp(-logits)))
                # When transforming log-odds to probabilities, there will be
                # some loss of precision. Besides, the log_prob may become
                # very large. So the absolute tolerance (atol) can't be the
                # default value (1e-06).
                self.assertAllClose(log_p.eval(), target_log_p, atol=0.01)
                p = binomial.prob(given)
                target_p = stats.binom.pmf(
                    given, n_experiments, 1 / (1. + np.exp(-logits)))
                self.assertAllClose(p.eval(), target_p, atol=0.01)

            _test_value(0., 6, [0, 1, 2, 3, 4, 5, 6])
            _test_value([5., -1., 5.], 2, [0, 0, 1])
            _test_value([10., -10., 0.], 200,
                        [[10, 10, 10], [190, 190, 190]])
            _test_value([[1., 5., 10.], [-1., -5., -10.]], 20,
                        np.ones([3, 1, 2, 3], dtype=np.int32))
            _test_value([[1., 5., 10.], [-1., -5., -10.]], 20,
                        19 * np.ones([3, 1, 2, 3], dtype=np.int32))

    def test_check_numerics(self):
        logits = tf.placeholder(tf.float32, [])
        given = tf.placeholder(tf.int32, [])
        binomial = Binomial(logits, 10, check_numerics=True)
        log_p = binomial.log_prob(given)
        with self.test_session(use_gpu=True):
            with self.assertRaisesRegexp(
                    tf.errors.InvalidArgumentError,
                    "lgamma\(given \+ 1\).*Tensor had Inf"):
                log_p.eval(feed_dict={logits: 1., given: -2})
            with self.assertRaisesRegexp(
                    tf.errors.InvalidArgumentError,
                    "lgamma\(n - given \+ 1\).*Tensor had Inf"):
                log_p.eval(feed_dict={logits: 1., given: 12})

    def test_dtype(self):
        def _distribution(param, dtype=None):
            return Binomial(param, 10, dtype)
        utils.test_dtype_1parameter_discrete(self, _distribution)

        with self.assertRaisesRegexp(TypeError, "n_experiments must be"):
            Binomial(1., tf.placeholder(tf.float32, []))


class TestInverseGamma(tf.test.TestCase):
    def test_init_check_shape(self):
        with self.test_session(use_gpu=True):
            with self.assertRaisesRegexp(ValueError,
                                         "should be broadcastable to match"):
                InverseGamma(alpha=tf.ones([2, 1]), beta=tf.ones([2, 4, 3]))

        InverseGamma(tf.placeholder(tf.float32, [None, 1]),
                     tf.placeholder(tf.float32, [None, 1, 3]))

    def test_value_shape(self):
        # static
        inv_gamma = InverseGamma(alpha=tf.placeholder(tf.float32, None),
                                 beta=tf.placeholder(tf.float32, None))
        self.assertEqual(inv_gamma.get_value_shape().as_list(), [])

        # dynamic
        with self.test_session(use_gpu=True):
            self.assertEqual(inv_gamma._value_shape().eval().tolist(), [])
        self.assertEqual(inv_gamma._value_shape().dtype, tf.int32)

    def test_batch_shape(self):
        utils.test_batch_shape_2parameter_univariate(
            self, InverseGamma, np.ones, np.ones)

    def test_sample_shape(self):
        utils.test_sample_shape_2parameter_univariate(
            self, InverseGamma, np.ones, np.ones)

    def test_log_prob_shape(self):
        utils.test_log_prob_shape_2parameter_univariate(
            self, InverseGamma, np.ones, np.ones, np.ones)

    def test_value(self):
        with self.test_session(use_gpu=True):
            def _test_value(alpha, beta, given):
                alpha = np.array(alpha, np.float32)
                beta = np.array(beta, np.float32)
                given = np.array(given, np.float32)
                inv_gamma = InverseGamma(alpha, beta)
                log_p = inv_gamma.log_prob(given)
                target_log_p = stats.invgamma.logpdf(given, alpha, scale=beta)
                self.assertAllClose(log_p.eval(), target_log_p)
                p = inv_gamma.prob(given)
                target_p = stats.invgamma.pdf(given, alpha, scale=beta)
                self.assertAllClose(p.eval(), target_p)

            _test_value(1., 1., [1., 10., 1e8])
            _test_value([0.5, 1., 2., 3., 5., 7.5, 9.],
                        [2., 2., 2., 1., 0.5, 1., 1.],
                        np.transpose([np.arange(1, 20)]))
            _test_value([1e-8, 1e8], [[1., 1e8], [1e-8, 5.]], [7.])

    def test_check_numerics(self):
        alpha = tf.placeholder(tf.float32, [])
        beta = tf.placeholder(tf.float32, [])
        given = tf.placeholder(tf.float32, [])
        inv_gamma = InverseGamma(alpha, beta, check_numerics=True)
        log_p = inv_gamma.log_prob(given)
        with self.test_session(use_gpu=True):
            with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                         "log\(given\).*Tensor had Inf"):
                log_p.eval(feed_dict={alpha: 1., beta: 1., given: 0.})
            with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                         "log\(beta\).*Tensor had NaN"):
                log_p.eval(feed_dict={alpha: 1., beta: -1., given: 1.})
            with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                         "log\(alpha\).*Tensor had NaN"):
                log_p.eval(feed_dict={alpha: -1.5, beta: 1., given: 1.})

    def test_dtype(self):
        utils.test_dtype_2parameter(self, InverseGamma)


class TestLaplace(tf.test.TestCase):
    def test_init_check_shape(self):
        with self.test_session(use_gpu=True):
            with self.assertRaisesRegexp(ValueError,
                                         "should be broadcastable to match"):
                Laplace(loc=tf.ones([2, 1]), scale=tf.ones([2, 4, 3]))

        Laplace(tf.placeholder(tf.float32, [None, 1]),
                tf.placeholder(tf.float32, [None, 1, 3]))

    def test_value_shape(self):
        # static
        laplace = Laplace(loc=tf.placeholder(tf.float32, None),
                          scale=tf.placeholder(tf.float32, None))
        self.assertEqual(laplace.get_value_shape().as_list(), [])

        # dynamic
        with self.test_session(use_gpu=True):
            self.assertEqual(laplace._value_shape().eval().tolist(), [])
        self.assertEqual(laplace._value_shape().dtype, tf.int32)

    def test_batch_shape(self):
        utils.test_batch_shape_2parameter_univariate(
            self, Laplace, np.zeros, np.ones)

    def test_sample_shape(self):
        utils.test_sample_shape_2parameter_univariate(
            self, Laplace, np.zeros, np.ones)

    def test_sample_reparameterized(self):
        loc = tf.ones([2, 3])
        scale = tf.ones([2, 3])
        laplace_rep = Laplace(loc, scale)
        samples = laplace_rep.sample(tf.placeholder(tf.int32, shape=[]))
        loc_grads, scale_grads = tf.gradients(samples, [loc, scale])
        self.assertTrue(loc_grads is not None)
        self.assertTrue(scale_grads is not None)

        laplace_no_rep = Laplace(loc, scale, is_reparameterized=False)
        samples = laplace_no_rep.sample(tf.placeholder(tf.int32, shape=[]))
        loc_grads, scale_grads = tf.gradients(samples, [loc, scale])
        self.assertEqual(loc_grads, None)
        self.assertEqual(scale_grads, None)

    def test_log_prob_shape(self):
        utils.test_log_prob_shape_2parameter_univariate(
            self, Laplace, np.zeros, np.ones, np.zeros)

    def test_value(self):
        with self.test_session(use_gpu=True):
            def _test_value(loc, scale, given):
                loc = np.array(loc, np.float32)
                scale = np.array(scale, np.float32)
                given = np.array(given, np.float32)
                laplace = Laplace(loc, scale)
                log_p = laplace.log_prob(given)
                target_log_p = stats.laplace.logpdf(given, loc, scale=scale)
                self.assertAllClose(log_p.eval(), target_log_p)
                p = laplace.prob(given)
                target_p = stats.laplace.pdf(given, loc, scale=scale)
                self.assertAllClose(p.eval(), target_p)

            _test_value(0., 1., [.01, .1, 1., 10., 100.])
            _test_value([-3, -2, -1, 0, 1, 2, 3],
                        [.1, 3, 2, 3, 3, 2, .1],
                        np.transpose([np.arange(1, 20)]))
            _test_value([1e-5, -1e-5], [[1., 10.], [1e8, 5.]], [7.])

    def test_check_numerics(self):
        loc = tf.placeholder(tf.float32, [])
        scale = tf.placeholder(tf.float32, [])
        given = tf.placeholder(tf.float32, [])
        laplace = Laplace(loc, scale, check_numerics=True)
        log_p = laplace.log_prob(given)
        with self.test_session(use_gpu=True):
            with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                         "log\(scale\).*Tensor had NaN"):
                log_p.eval(feed_dict={loc: 1., scale: -1., given: 1.})

    def test_dtype(self):
        utils.test_dtype_2parameter(self, Laplace)
