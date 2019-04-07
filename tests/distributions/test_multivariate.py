#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from contextlib import contextmanager
import tensorflow as tf
import numpy as np
from scipy import stats
from scipy.special import logsumexp, factorial, gammaln

from tests.distributions import utils
from zhusuan.distributions.multivariate import *


class TestMultivariateNormalCholesky(tf.test.TestCase):
    def test_init_check_shape(self):
        with self.session(use_gpu=True):
            with self.assertRaisesRegexp(ValueError, "should have rank"):
                MultivariateNormalCholesky(tf.zeros([]), tf.zeros([]))
            with self.assertRaisesRegexp(ValueError, "should have rank"):
                MultivariateNormalCholesky(tf.zeros([1]), tf.zeros([1]))
            with self.assertRaisesRegexp(ValueError, 'compatible'):
                MultivariateNormalCholesky(
                    tf.zeros([1, 2]), tf.placeholder(tf.float32, [1, 2, 3]))
            u = tf.placeholder(tf.float32, [None])
            len_u = tf.shape(u)[0]
            dst = MultivariateNormalCholesky(
                tf.zeros([2]), tf.zeros([len_u, len_u]))
            with self.assertRaisesRegexp(
                    tf.errors.InvalidArgumentError, 'compatible'):
                dst.sample().eval(feed_dict={u: np.ones((3,))})

    def test_shape_inference(self):
        with self.session(use_gpu=True):
            # Static
            mean = 10 * np.random.normal(size=(10, 11, 2)).astype('d')
            cov = np.zeros((10, 11, 2, 2))
            dst = MultivariateNormalCholesky(
                tf.constant(mean), tf.constant(cov))
            self.assertEqual(dst.get_batch_shape().as_list(), [10, 11])
            self.assertEqual(dst.get_value_shape().as_list(), [2])
            # Dynamic
            unk_mean = tf.placeholder(tf.float32, None)
            unk_cov = tf.placeholder(tf.float32, None)
            dst = MultivariateNormalCholesky(unk_mean, unk_cov)
            self.assertEqual(dst.get_value_shape().as_list(), [None])
            feed_dict = {unk_mean: np.ones(2), unk_cov: np.eye(2)}
            self.assertEqual(list(dst.batch_shape.eval(feed_dict)), [])
            self.assertEqual(list(dst.value_shape.eval(feed_dict)), [2])

    def _gen_test_params(self, seed):
        np.random.seed(seed)
        mean = 10 * np.random.normal(size=(10, 11, 3)).astype('d')
        cov = np.zeros((10, 11, 3, 3))
        cov_chol = np.zeros_like(cov)
        for i in range(10):
            for j in range(11):
                cov[i, j] = stats.invwishart.rvs(3, np.eye(3))
                cov[i, j] /= np.max(np.diag(cov[i, j]))
                cov_chol[i, j, :, :] = np.linalg.cholesky(cov[i, j])
        return mean, cov, cov_chol

    @contextmanager
    def fixed_randomness_session(self, seed):
        with tf.Graph().as_default() as g:
            with self.session(use_gpu=True, graph=g):
                tf.set_random_seed(seed)
                yield

    def test_sample(self):
        with self.fixed_randomness_session(233):
            def test_sample_with(seed):
                mean, cov, cov_chol = self._gen_test_params(seed)
                dst = MultivariateNormalCholesky(
                    tf.constant(mean), tf.constant(cov_chol))
                n_exp = 20000
                samples = dst.sample(n_exp)
                sample_shape = (n_exp, 10, 11, 3)
                self.assertEqual(samples.shape.as_list(), list(sample_shape))
                samples = dst.sample(n_exp).eval()
                self.assertEqual(samples.shape, sample_shape)
                self.assertAllClose(
                    np.mean(samples, axis=0), mean, rtol=5e-2, atol=5e-2)
                for i in range(10):
                    for j in range(11):
                        self.assertAllClose(
                            np.cov(samples[:, i, j, :].T), cov[i, j],
                            rtol=1e-1, atol=1e-1)

            for seed in [23, 233, 2333]:
                test_sample_with(seed)

    def test_prob(self):
        with self.fixed_randomness_session(233):
            def test_prob_with(seed):
                mean, cov, cov_chol = self._gen_test_params(seed)
                dst = MultivariateNormalCholesky(
                    tf.constant(mean), tf.constant(cov_chol),
                    check_numerics=True)
                n_exp = 200
                samples = dst.sample(n_exp).eval()
                log_pdf = dst.log_prob(tf.constant(samples))
                pdf_shape = (n_exp, 10, 11)
                self.assertEqual(log_pdf.shape.as_list(), list(pdf_shape))
                log_pdf = log_pdf.eval()
                self.assertEqual(log_pdf.shape, pdf_shape)
                for i in range(10):
                    for j in range(11):
                        log_pdf_exact = stats.multivariate_normal.logpdf(
                                samples[:, i, j, :], mean[i, j], cov[i, j])
                        self.assertAllClose(
                            log_pdf_exact, log_pdf[:, i, j])
                self.assertAllClose(
                    np.exp(log_pdf), dst.prob(tf.constant(samples)).eval())

            for seed in [23, 233, 2333]:
                test_prob_with(seed)

    def test_sample_reparameterized(self):
        mean, cov, cov_chol = self._gen_test_params(23)
        mean = tf.constant(mean)
        cov_chol = tf.constant(cov_chol)
        mvn_rep = MultivariateNormalCholesky(mean, cov_chol)
        samples = mvn_rep.sample(tf.placeholder(tf.int32, shape=[]))
        mean_grads, cov_grads = tf.gradients(samples, [mean, cov_chol])
        self.assertTrue(mean_grads is not None)
        self.assertTrue(cov_grads is not None)
        mvn_no_rep = MultivariateNormalCholesky(
                mean, cov_chol, is_reparameterized=False)
        samples = mvn_no_rep.sample(tf.placeholder(tf.int32, shape=[]))
        mean_grads, cov_grads = tf.gradients(samples, [mean, cov_chol])
        self.assertEqual(mean_grads, None)
        self.assertEqual(cov_grads, None)


class TestMultinomial(tf.test.TestCase):
    def test_init_check_shape(self):
        with self.session(use_gpu=True):
            with self.assertRaisesRegexp(ValueError, "should have rank"):
                Multinomial(tf.zeros([]), n_experiments=10)

    def test_init_n(self):
        dist = Multinomial(tf.ones([2]), n_experiments=10)
        self.assertTrue(isinstance(dist.n_categories, int))
        self.assertEqual(dist.n_categories, 2)
        self.assertTrue(isinstance(dist.n_experiments, int))
        self.assertEqual(dist.n_experiments, 10)

        with self.session(use_gpu=True) as sess:
            logits = tf.placeholder(tf.float32, None)
            n_experiments = tf.placeholder(tf.int32, None)
            dist2 = Multinomial(logits, n_experiments=n_experiments)
            self.assertEqual(
                sess.run([dist2.n_categories, dist2.n_experiments],
                         feed_dict={logits: np.ones([2]), n_experiments: 10}),
                [2, 10])
            with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                         "should have rank"):
                dist2.n_categories.eval(feed_dict={logits: 1.,
                                                   n_experiments: 10})
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
        dist = Multinomial(tf.placeholder(tf.float32, [None, 2]),
                           n_experiments=10)
        self.assertEqual(dist.get_value_shape().as_list(), [2])

        # dynamic
        logits = tf.placeholder(tf.float32, None)
        dist2 = Multinomial(logits, n_experiments=10)
        self.assertTrue(dist2._value_shape().dtype is tf.int32)
        with self.session(use_gpu=True):
            self.assertEqual(dist2._value_shape().eval(
                feed_dict={logits: np.ones([2])}).tolist(), [2])

        self.assertEqual(dist._value_shape().dtype, tf.int32)

    def test_batch_shape(self):
        def _distribution(param):
            return Multinomial(param, n_experiments=10)
        utils.test_batch_shape_1parameter(
            self, _distribution, np.zeros, is_univariate=False)

    def test_sample(self):
        def _distribution(param):
            return Multinomial(param, n_experiments=10)
        utils.test_1parameter_sample_shape_one_rank_less(
            self, _distribution, np.zeros)
        dist = Multinomial(np.ones([2, 2]), n_experiments=None)
        with self.assertRaisesRegexp(ValueError,
                                     "Cannot sample when `n_experiments`"):
            dist.sample()

    def test_log_prob_shape(self):
        def _distribution(param):
            return Multinomial(param, n_experiments=10)

        def _make_samples(shape):
            samples = np.zeros(shape)
            samples = samples.reshape((-1, shape[-1]))
            samples[:, 0] = 1
            return samples.reshape(shape)

        utils.test_1parameter_log_prob_shape_one_rank_less(
            self, _distribution, _make_samples, _make_samples)

    def test_value(self):
        with self.session(use_gpu=True):
            def _test_value(logits, n_experiments, given, normalize_logits):
                logits = np.array(logits, np.float32)
                given = np.array(given)
                dist = Multinomial(logits, n_experiments=None,
                                   normalize_logits=normalize_logits)
                log_p = dist.log_prob(given)
                if n_experiments is not None:
                    dist2 = Multinomial(logits, n_experiments=n_experiments,
                                        normalize_logits=normalize_logits)
                    log_p_2 = dist2.log_prob(given)
                    self.assertAllClose(log_p.eval(), log_p_2.eval())

                maybe_normalized_logits = logits
                if normalize_logits:
                    maybe_normalized_logits -= logsumexp(
                        logits, axis=-1, keepdims=True)
                n_experiments = np.sum(given, axis=-1)
                target_log_p = np.log(factorial(n_experiments)) - \
                    np.sum(np.log(factorial(given)), -1) + \
                    np.sum(given * maybe_normalized_logits, -1)
                self.assertAllClose(log_p.eval(), target_log_p)
                p = dist.prob(given)
                target_p = np.exp(target_log_p)
                self.assertAllClose(np.log(p.eval()), np.log(target_p))

            for normalize_logits in [True, False]:
                _test_value([-50., -20., 0.], 4, [1, 0, 3], normalize_logits)
                _test_value([1., 10., 1000.], 1, [1, 0, 0], normalize_logits)
                _test_value([[2., 3., 1.], [5., 7., 4.]], 7,
                            np.array([3, 1, 3], dtype=np.int32),
                            normalize_logits)
                _test_value([-10., 10., 20., 50.], 100,
                            [[0, 1, 49, 50], [50, 49, 1, 0]],
                            normalize_logits)

    def test_dtype(self):
        def _distribution(param, **kwargs):
            return Multinomial(param, n_experiments=10, **kwargs)
        utils.test_dtype_1parameter_discrete(self, _distribution)

        with self.assertRaisesRegexp(TypeError, "n_experiments must be"):
            Multinomial([1., 1.], n_experiments=tf.placeholder(tf.float32, []))
        with self.assertRaisesRegexp(TypeError,
                                     "n_experiments must be integer"):
            Multinomial([1., 1.], n_experiments=2.0)


class TestUnnormalizedMultinomial(tf.test.TestCase):
    def test_init_check_shape(self):
        with self.session(use_gpu=True):
            with self.assertRaisesRegexp(ValueError, "should have rank"):
                UnnormalizedMultinomial(tf.zeros([]))

    def test_init_n(self):
        dist = UnnormalizedMultinomial(tf.ones([2]))
        self.assertTrue(isinstance(dist.n_categories, int))
        self.assertEqual(dist.n_categories, 2)

        with self.session(use_gpu=True) as sess:
            logits = tf.placeholder(tf.float32, None)
            dist2 = UnnormalizedMultinomial(logits)
            self.assertEqual(
                sess.run(dist2.n_categories, feed_dict={logits: np.ones([2])}),
                2)
            with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                         "should have rank"):
                dist2.n_categories.eval(feed_dict={logits: 1.})

    def test_value_shape(self):
        # static
        dist = UnnormalizedMultinomial(tf.placeholder(tf.float32, [None, 2]))
        self.assertEqual(dist.get_value_shape().as_list(), [2])

        # dynamic
        logits = tf.placeholder(tf.float32, None)
        dist2 = UnnormalizedMultinomial(logits)
        self.assertTrue(dist2._value_shape().dtype is tf.int32)
        with self.session(use_gpu=True):
            self.assertEqual(dist2._value_shape().eval(
                feed_dict={logits: np.ones([2])}).tolist(), [2])

        self.assertEqual(dist._value_shape().dtype, tf.int32)

    def test_batch_shape(self):
        utils.test_batch_shape_1parameter(
            self, UnnormalizedMultinomial, np.zeros, is_univariate=False)

    def test_sample(self):
        dist = UnnormalizedMultinomial(np.ones([2, 2]))
        with self.assertRaisesRegexp(NotImplementedError,
                                     "Unnormalized multinomial distribution"
                                     " does not support sampling"):
            dist.sample()

    def test_log_prob_shape(self):
        def _distribution(param):
            return UnnormalizedMultinomial(param)

        def _make_samples(shape):
            samples = np.zeros(shape)
            samples = samples.reshape((-1, shape[-1]))
            samples[:, 0] = 1
            return samples.reshape(shape)

        utils.test_1parameter_log_prob_shape_one_rank_less(
            self, _distribution, _make_samples, _make_samples)

    def test_value(self):
        with self.session(use_gpu=True):
            def _test_value(logits, given, normalize_logits):
                logits = np.array(logits, np.float32)
                given = np.array(given)
                dist = UnnormalizedMultinomial(
                    logits, normalize_logits=normalize_logits)
                log_p = dist.log_prob(given)

                maybe_normalized_logits = logits
                if normalize_logits:
                    maybe_normalized_logits -= logsumexp(
                        logits, axis=-1, keepdims=True)
                target_log_p = np.sum(given * maybe_normalized_logits, -1)
                self.assertAllClose(log_p.eval(), target_log_p)
                p = dist.prob(given)
                target_p = np.exp(target_log_p)
                self.assertAllClose(p.eval(), target_p)

            for normalize_logits in [True, False]:
                _test_value([-50., -20., 0.], [1, 0, 3], normalize_logits)
                _test_value([1., 10., 1000.], [1, 0, 0], normalize_logits)
                _test_value([[2., 3., 1.], [5., 7., 4.]],
                            np.ones([3, 1, 3], dtype=np.int32),
                            normalize_logits)
                _test_value([-10., 10., 20., 50.],
                            [[0, 1, 99, 100], [100, 99, 1, 0]],
                            normalize_logits)

    def test_dtype(self):
        utils.test_dtype_1parameter_discrete(self, UnnormalizedMultinomial,
                                             prob_only=True)


class TestOnehotCategorical(tf.test.TestCase):
    def test_init_check_shape(self):
        with self.session(use_gpu=True):
            with self.assertRaisesRegexp(ValueError, "should have rank"):
                OnehotCategorical(logits=tf.zeros([]))

    def test_init_n_categories(self):
        cat = OnehotCategorical(tf.ones([10]))
        self.assertTrue(isinstance(cat.n_categories, int))
        self.assertEqual(cat.n_categories, 10)

        with self.session(use_gpu=True):
            logits = tf.placeholder(tf.float32, None)
            cat2 = OnehotCategorical(logits)
            self.assertEqual(
                cat2.n_categories.eval(feed_dict={logits: np.ones([10])}), 10)
            with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                         "should have rank"):
                cat2.n_categories.eval(feed_dict={logits: 1.})

    def test_value_shape(self):
        # static
        cat = OnehotCategorical(tf.placeholder(tf.float32, [None, 10]))
        self.assertEqual(cat.get_value_shape().as_list(), [10])

        # dynamic
        logits = tf.placeholder(tf.float32, None)
        cat2 = OnehotCategorical(logits)
        self.assertTrue(cat2._value_shape().dtype is tf.int32)
        with self.session(use_gpu=True):
            self.assertEqual(cat2._value_shape().eval(
                feed_dict={logits: np.ones([2, 1, 3])}).tolist(), [3])

        self.assertEqual(cat._value_shape().dtype, tf.int32)

    def test_batch_shape(self):
        utils.test_batch_shape_1parameter(
            self, OnehotCategorical, np.zeros, is_univariate=False)

    def test_sample_shape(self):
        utils.test_1parameter_sample_shape_one_rank_less(
            self, OnehotCategorical, np.zeros)

    def test_log_prob_shape(self):
        def _make_samples(shape):
            samples = np.zeros(shape)
            samples = samples.reshape((-1, shape[-1]))
            samples[:, 0] = 1
            return samples.reshape(shape)

        utils.test_1parameter_log_prob_shape_one_rank_less(
            self, OnehotCategorical, _make_samples, _make_samples)

    def test_value(self):
        with self.session(use_gpu=True):
            def _test_value(logits, given):
                logits = np.array(logits, np.float32)
                normalized_logits = logits - logsumexp(
                    logits, axis=-1, keepdims=True)
                given = np.array(given, np.int32)
                cat = OnehotCategorical(logits)
                log_p = cat.log_prob(tf.one_hot(given, logits.shape[-1],
                                                dtype=tf.int32))

                def _one_hot(x, depth):
                    n_elements = x.size
                    ret = np.zeros((n_elements, depth))
                    ret[np.arange(n_elements), x.flat] = 1
                    return ret.reshape(list(x.shape) + [depth])

                target_log_p = np.sum(_one_hot(
                    given, logits.shape[-1]) * normalized_logits, -1)
                self.assertAllClose(log_p.eval(), target_log_p)
                p = cat.prob(tf.one_hot(given, logits.shape[-1],
                                        dtype=tf.int32))
                target_p = np.sum(_one_hot(
                    given, logits.shape[-1]) * np.exp(normalized_logits), -1)
                self.assertAllClose(p.eval(), target_p)

            _test_value([0.], [0, 0, 0])
            _test_value([-50., -10., -50.], [0, 1, 2, 1])
            _test_value([0., 4.], [[0, 1], [0, 1]])
            _test_value([[2., 3., 1.], [5., 7., 4.]],
                        np.ones([3, 1, 1], dtype=np.int32))

    def test_dtype(self):
        utils.test_dtype_1parameter_discrete(self, OnehotCategorical)


class TestDirichlet(tf.test.TestCase):
    def test_init_check_shape(self):
        with self.session(use_gpu=True):
            with self.assertRaisesRegexp(ValueError, "should have rank"):
                Dirichlet(alpha=tf.zeros([]))

    def test_init_n_categories(self):
        dist = Dirichlet(tf.ones([10]))
        self.assertTrue(isinstance(dist.n_categories, int))
        self.assertEqual(dist.n_categories, 10)
        with self.assertRaisesRegexp(ValueError,
                                     "n_categories.*should be at least 2"):
            Dirichlet(tf.ones([3, 1]))
        dist2 = Dirichlet(tf.placeholder(tf.float32, [3, None]))
        self.assertTrue(dist2.n_categories is not None)

        with self.session(use_gpu=True):
            alpha = tf.placeholder(tf.float32, None)
            dist3 = Dirichlet(alpha)
            self.assertEqual(
                dist3.n_categories.eval(feed_dict={alpha: np.ones([10])}), 10)
            with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                         "should have rank"):
                dist3.n_categories.eval(feed_dict={alpha: 1.})

    def test_value_shape(self):
        # static
        dist = Dirichlet(tf.placeholder(tf.float32, [None, 10]))
        self.assertEqual(dist.get_value_shape().as_list(), [10])

        # dynamic
        alpha = tf.placeholder(tf.float32, None)
        dist2 = Dirichlet(alpha)
        self.assertEqual(dist2.get_value_shape().as_list(), [None])
        self.assertTrue(dist2._value_shape().dtype is tf.int32)
        with self.session(use_gpu=True):
            self.assertEqual(dist2._value_shape().eval(
                feed_dict={alpha: np.ones([2, 1, 3])}).tolist(), [3])
        self.assertEqual(dist._value_shape().dtype, tf.int32)

    def test_batch_shape(self):
        utils.test_batch_shape_1parameter(
            self, Dirichlet, np.zeros, is_univariate=False)

    def test_sample_shape(self):
        utils.test_1parameter_sample_shape_one_rank_less(
            self, Dirichlet, np.zeros)

    def test_log_prob_shape(self):
        def _make_samples(shape):
            samples = np.ones(shape, dtype=np.float32)
            return samples / samples.sum(axis=-1, keepdims=True)

        # TODO: This failed with a bug in Tensorflow, waiting fix.
        # https://github.com/tensorflow/tensorflow/issues/8391
        # _test_static([3, None], [3, 2, 1, None], [3, 2, 3])
        utils.test_1parameter_log_prob_shape_one_rank_less(
            self, Dirichlet, np.ones, _make_samples)

    def test_value(self):
        def dirichlet_logpdf(x, alpha):
            # scipy's implementation of dirichlet logpdf doesn't support
            # batch of x, we use this modified version.
            def _lnB(alpha):
                return np.sum(gammaln(alpha)) - gammaln(np.sum(alpha))

            lnB = _lnB(alpha)
            return - lnB + np.sum(np.log(x) * (alpha - 1), -1)

        def dirichlet_pdf(x, alpha):
            return np.exp(dirichlet_logpdf(x, alpha))

        with self.session(use_gpu=True):
            def _test_value_alpha_rank1(alpha, given):
                alpha = np.array(alpha, np.float32)
                given = np.array(given, np.float32)
                dist = Dirichlet(alpha)
                log_p = dist.log_prob(given)
                target_log_p = dirichlet_logpdf(given, alpha)
                self.assertAllClose(log_p.eval(), target_log_p)
                p = dist.prob(given)
                target_p = dirichlet_pdf(given, alpha)
                self.assertAllClose(p.eval(), target_p)

            _test_value_alpha_rank1([1., 1., 1.],
                                    [[0.2, 0.5, 0.3], [0.3, 0.4, 0.3]])
            _test_value_alpha_rank1([2., 3., 4.], [0.3, 0.7, 0.])
            # TODO: fix for case when alpha=1, given=0

            def _test_value_alpha_rank2_given_rank2(alpha, given):
                alpha = np.array(alpha, np.float32)
                given = np.array(given, np.float32)
                alpha_b = alpha * np.ones_like(given)
                given_b = given * np.ones_like(alpha)
                dist = Dirichlet(alpha)
                log_p = dist.log_prob(given)
                target_log_p = np.array(
                    [dirichlet_logpdf(given_b[i], alpha_b[i])
                     for i in range(alpha_b.shape[0])])
                self.assertAllClose(log_p.eval(), target_log_p)
                p = dist.prob(given)
                target_p = np.array(
                    [dirichlet_pdf(given_b[i], alpha_b[i])
                     for i in range(alpha_b.shape[0])])
                self.assertAllClose(p.eval(), target_p)

            _test_value_alpha_rank2_given_rank2([[1., 2.], [3., 4.]],
                                                [0.5, 0.5])
            _test_value_alpha_rank2_given_rank2([[5., 6.], [7., 8.]],
                                                [[0.1, 0.9]])
            _test_value_alpha_rank2_given_rank2([[100., 1.], [0.01, 10.]],
                                                [[0., 1.], [1., 0.]])

    def test_check_numerics(self):
        alpha = tf.placeholder(tf.float32, None)
        given = tf.placeholder(tf.float32, None)
        dist = Dirichlet(alpha, check_numerics=True)
        log_p = dist.log_prob(given)
        with self.session(use_gpu=True):
            with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                         "log\(given\).*Tensor had Inf"):
                log_p.eval(feed_dict={alpha: np.ones([2]), given: [0., 1.]})
            with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                         "lbeta\(alpha\).*Tensor had NaN"):
                log_p.eval(feed_dict={alpha: [-1., 1.], given: [0.5, 0.5]})

    def test_dtype(self):
        utils.test_dtype_1parameter_continuous(self, Dirichlet)


class TestExpConcrete(tf.test.TestCase):
    def test_init_check_shape(self):
        with self.session(use_gpu=True):
            with self.assertRaisesRegexp(ValueError, "should have rank"):
                ExpConcrete(1., logits=tf.zeros([]))

    def test_init_n_categories(self):
        con = ExpConcrete(1., tf.ones([10]))
        self.assertTrue(isinstance(con.n_categories, int))
        self.assertEqual(con.n_categories, 10)

        with self.session(use_gpu=True):
            logits = tf.placeholder(tf.float32, None)
            con2 = ExpConcrete(1., logits)
            self.assertEqual(
                con2.n_categories.eval(feed_dict={logits: np.ones([10])}), 10)
            with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                         "should have rank"):
                con2.n_categories.eval(feed_dict={logits: 1.})

    def test_init_temperature(self):
        with self.assertRaisesRegexp(ValueError,
                                     "should be a scalar"):
            ExpConcrete([1.], [1., 2.])

        with self.session(use_gpu=True):
            temperature = tf.placeholder(tf.float32, None)
            con = ExpConcrete(temperature, [1., 2.])
            with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                         "should be a scalar"):
                con.temperature.eval(feed_dict={temperature: [1.]})

    def test_value_shape(self):
        # static
        con = ExpConcrete(1., tf.placeholder(tf.float32, [None, 10]))
        self.assertEqual(con.get_value_shape().as_list(), [10])

        # dynamic
        logits = tf.placeholder(tf.float32, None)
        con2 = ExpConcrete(1., logits)
        self.assertTrue(con2._value_shape().dtype is tf.int32)
        with self.session(use_gpu=True):
            self.assertEqual(con2._value_shape().eval(
                feed_dict={logits: np.ones([2, 1, 3])}).tolist(), [3])

        self.assertEqual(con._value_shape().dtype, tf.int32)

    def test_batch_shape(self):
        def _proxy_distribution(logits):
            return ExpConcrete(1., logits)
        utils.test_batch_shape_1parameter(
            self, _proxy_distribution, np.zeros, is_univariate=False)

    def test_sample_shape(self):
        def _proxy_distribution(logits):
            return ExpConcrete(1., logits)
        utils.test_1parameter_sample_shape_one_rank_less(
            self, _proxy_distribution, np.zeros)

    def test_log_prob_shape(self):
        def _proxy_distribution(logits):
            return ExpConcrete(1., logits)

        def _make_samples(shape):
            samples = np.ones(shape, dtype=np.float32)
            return np.log(samples / samples.sum(axis=-1, keepdims=True))

        utils.test_1parameter_log_prob_shape_one_rank_less(
            self, _proxy_distribution, np.ones, _make_samples)

    def test_value(self):
        with self.session(use_gpu=True):
            def _test_value(given, temperature, logits):
                given = np.array(given, np.float32)
                logits = np.array(logits, np.float32)
                n = logits.shape[-1]

                t = temperature
                target_log_p = gammaln(n) + (n - 1) * np.log(t) + \
                    (logits - t * given).sum(axis=-1) - \
                    n * np.log(np.exp(logits - t * given).sum(axis=-1))

                con = ExpConcrete(temperature, logits=logits)
                log_p = con.log_prob(given)
                self.assertAllClose(log_p.eval(), target_log_p)
                p = con.prob(given)
                self.assertAllClose(p.eval(), np.exp(target_log_p))

            _test_value([np.log(0.25), np.log(0.25), np.log(0.5)],
                        0.1,
                        [1., 1., 1.2])
            _test_value([[np.log(0.25), np.log(0.25), np.log(0.5)],
                        [np.log(0.1), np.log(0.5), np.log(0.4)]],
                        0.5,
                        [[1., 1., 1.], [.5, .5, .4]])

    def test_dtype(self):
        utils.test_dtype_2parameter(self, ExpConcrete)

    def test_sample_reparameterized(self):
        temperature = tf.constant(1.0)
        logits = tf.ones([2, 3])
        con_rep = ExpConcrete(temperature, logits)
        samples = con_rep.sample(tf.placeholder(tf.int32, shape=[]))
        t_grads, logits_grads = tf.gradients(samples, [temperature, logits])
        self.assertTrue(t_grads is not None)
        self.assertTrue(logits_grads is not None)

        con_no_rep = ExpConcrete(temperature, logits, is_reparameterized=False)
        samples = con_no_rep.sample(tf.placeholder(tf.int32, shape=[]))
        t_grads, logits_grads = tf.gradients(samples, [temperature, logits])
        self.assertEqual(t_grads, None)
        self.assertEqual(logits_grads, None)

    def test_path_derivative(self):
        temperature = tf.constant(1.0)
        logits = tf.ones([2, 3])
        n_samples = tf.placeholder(tf.int32, shape=[])

        con_rep = ExpConcrete(temperature, logits, use_path_derivative=True)
        samples = con_rep.sample(n_samples)
        log_prob = con_rep.log_prob(samples)
        t_path_grads, logits_path_grads = tf.gradients(log_prob,
                                                       [temperature, logits])
        sample_grads = tf.gradients(log_prob, samples)
        t_true_grads = tf.gradients(samples, temperature, sample_grads)[0]
        logits_true_grads = tf.gradients(samples, logits, sample_grads)[0]
        with self.session(use_gpu=True) as sess:
            outs = sess.run([t_path_grads, t_true_grads,
                             logits_path_grads, logits_true_grads],
                            feed_dict={n_samples: 7})
            t_path, t_true, logits_path, logits_true = outs
            self.assertAllClose(t_path, t_true)
            self.assertAllClose(logits_path, logits_true)

        con_no_rep = ExpConcrete(temperature, logits, is_reparameterized=False,
                                 use_path_derivative=True)
        samples = con_no_rep.sample(n_samples)
        log_prob = con_no_rep.log_prob(samples)
        t_path_grads, logits_path_grads = tf.gradients(log_prob,
                                                       [temperature, logits])
        self.assertTrue(t_path_grads is None)
        self.assertTrue(logits_path_grads is None)

    def test_check_numerics(self):
        tau = tf.placeholder(tf.float32, None)
        logits = tf.placeholder(tf.float32, None)
        given = tf.placeholder(tf.float32, None)
        dist = ExpConcrete(tau, logits, check_numerics=True)
        log_p = dist.log_prob(given)
        with self.session(use_gpu=True):
            with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                         "log\(temperature\).*Tensor had Inf"):
                log_p.eval(feed_dict={tau: 0., logits: np.ones([2]),
                                      given: [1., 1.]})
            with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                         "log\(temperature\).*Tensor had NaN"):
                log_p.eval(feed_dict={tau: -1., logits: np.ones([2]),
                                      given: [1., 1.]})


class TestConcrete(tf.test.TestCase):
    def test_init_check_shape(self):
        with self.session(use_gpu=True):
            with self.assertRaisesRegexp(ValueError, "should have rank"):
                Concrete(1., logits=tf.zeros([]))

    def test_init_n_categories(self):
        con = Concrete(1., tf.ones([10]))
        self.assertTrue(isinstance(con.n_categories, int))
        self.assertEqual(con.n_categories, 10)

        with self.session(use_gpu=True):
            logits = tf.placeholder(tf.float32, None)
            con2 = Concrete(1., logits)
            self.assertEqual(
                con2.n_categories.eval(feed_dict={logits: np.ones([10])}), 10)
            with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                         "should have rank"):
                con2.n_categories.eval(feed_dict={logits: 1.})

    def test_init_temperature(self):
        with self.assertRaisesRegexp(ValueError,
                                     "should be a scalar"):
            Concrete([1.], [1., 2.])

        with self.session(use_gpu=True):
            temperature = tf.placeholder(tf.float32, None)
            con = Concrete(temperature, [1., 2.])
            with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                         "should be a scalar"):
                con.temperature.eval(feed_dict={temperature: [1.]})

    def test_value_shape(self):
        # static
        con = Concrete(1., tf.placeholder(tf.float32, [None, 10]))
        self.assertEqual(con.get_value_shape().as_list(), [10])

        # dynamic
        logits = tf.placeholder(tf.float32, None)
        con2 = Concrete(1., logits)
        self.assertTrue(con2._value_shape().dtype is tf.int32)
        with self.session(use_gpu=True):
            self.assertEqual(con2._value_shape().eval(
                feed_dict={logits: np.ones([2, 1, 3])}).tolist(), [3])

        self.assertEqual(con._value_shape().dtype, tf.int32)

    def test_batch_shape(self):
        def _proxy_distribution(logits):
            return Concrete(1., logits)
        utils.test_batch_shape_1parameter(
            self, _proxy_distribution, np.zeros, is_univariate=False)

    def test_sample_shape(self):
        def _proxy_distribution(logits):
            return Concrete(1., logits)
        utils.test_1parameter_sample_shape_one_rank_less(
            self, _proxy_distribution, np.zeros)

    def test_log_prob_shape(self):
        def _proxy_distribution(logits):
            return Concrete(1., logits)

        def _make_samples(shape):
            samples = np.ones(shape, dtype=np.float32)
            return np.log(samples / samples.sum(axis=-1, keepdims=True))

        utils.test_1parameter_log_prob_shape_one_rank_less(
            self, _proxy_distribution, np.ones, _make_samples)

    def test_value(self):
        with self.session(use_gpu=True):
            def _test_value(given, temperature, logits):
                given = np.array(given, np.float32)
                logits = np.array(logits, np.float32)
                n = logits.shape[-1]

                t = temperature
                target_log_p = gammaln(n) + (n - 1) * np.log(t) + \
                    (logits - (t + 1) * np.log(given)).sum(axis=-1) - \
                    n * np.log(np.exp(logits - t * np.log(given)).sum(axis=-1))

                con = Concrete(temperature, logits=logits)
                log_p = con.log_prob(given)
                self.assertAllClose(log_p.eval(), target_log_p)
                p = con.prob(given)
                self.assertAllClose(p.eval(), np.exp(target_log_p))

            _test_value([0.25, 0.25, 0.5],
                        0.1,
                        [1., 1., 1.2])
            _test_value([[0.25, 0.25, 0.5],
                        [0.1, 0.5, 0.4]],
                        0.5,
                        [[1., 1., 1.], [.5, .5, .4]])

    def test_dtype(self):
        utils.test_dtype_2parameter(self, Concrete)

    def test_sample_reparameterized(self):
        temperature = tf.constant(1.0)
        logits = tf.ones([2, 3])
        con_rep = Concrete(temperature, logits)
        samples = con_rep.sample(tf.placeholder(tf.int32, shape=[]))
        t_grads, logits_grads = tf.gradients(samples, [temperature, logits])
        self.assertTrue(t_grads is not None)
        self.assertTrue(logits_grads is not None)

        con_no_rep = Concrete(temperature, logits, is_reparameterized=False)
        samples = con_no_rep.sample(tf.placeholder(tf.int32, shape=[]))
        t_grads, logits_grads = tf.gradients(samples, [temperature, logits])
        self.assertEqual(t_grads, None)
        self.assertEqual(logits_grads, None)

    def test_path_derivative(self):
        temperature = tf.constant(1.0)
        logits = tf.ones([2, 3])
        n_samples = tf.placeholder(tf.int32, shape=[])

        con_rep = Concrete(temperature, logits, use_path_derivative=True)
        samples = con_rep.sample(n_samples)
        log_prob = con_rep.log_prob(samples)
        t_path_grads, logits_path_grads = tf.gradients(log_prob,
                                                       [temperature, logits])
        sample_grads = tf.gradients(log_prob, samples)
        t_true_grads = tf.gradients(samples, temperature, sample_grads)[0]
        logits_true_grads = tf.gradients(samples, logits, sample_grads)[0]
        with self.session(use_gpu=True) as sess:
            outs = sess.run([t_path_grads, t_true_grads,
                             logits_path_grads, logits_true_grads],
                            feed_dict={n_samples: 7})
            t_path, t_true, logits_path, logits_true = outs
            self.assertAllClose(t_path, t_true)
            self.assertAllClose(logits_path, logits_true)

        con_no_rep = Concrete(temperature, logits, is_reparameterized=False,
                              use_path_derivative=True)
        samples = con_no_rep.sample(n_samples)
        log_prob = con_no_rep.log_prob(samples)
        t_path_grads, logits_path_grads = tf.gradients(log_prob,
                                                       [temperature, logits])
        self.assertTrue(t_path_grads is None)
        self.assertTrue(logits_path_grads is None)

    def test_check_numerics(self):
        tau = tf.placeholder(tf.float32, None)
        logits = tf.placeholder(tf.float32, None)
        given = tf.placeholder(tf.float32, None)
        dist = Concrete(tau, logits, check_numerics=True)
        log_p = dist.log_prob(given)
        with self.session(use_gpu=True):
            with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                         "log\(given\).*Tensor had Inf"):
                log_p.eval(feed_dict={tau: 1., logits: np.ones([2]),
                                      given: [0., 1.]})
            with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                         "log\(given\).*Tensor had NaN"):
                log_p.eval(feed_dict={tau: 1., logits: np.ones([2]),
                                      given: [1., -1.]})
            with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                         "log\(temperature\).*Tensor had Inf"):
                log_p.eval(feed_dict={tau: 0., logits: np.ones([2]),
                                      given: [1., 1.]})
            with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                         "log\(temperature\).*Tensor had NaN"):
                log_p.eval(feed_dict={tau: -1., logits: np.ones([2]),
                                      given: [1., 1.]})


class TestMatrixVariateNormalCholesky(tf.test.TestCase):
    def test_init_check_shape(self):
        with self.session(use_gpu=True):
            with self.assertRaisesRegexp(ValueError, "should have rank >= 2"):
                MatrixVariateNormalCholesky(
                    tf.zeros([]), tf.zeros([]), tf.zeros([]))
            with self.assertRaisesRegexp(ValueError, "should have rank >= 2"):
                MatrixVariateNormalCholesky(
                    tf.zeros([1, 2]), tf.zeros([1]), tf.zeros([2, 2]))
            with self.assertRaisesRegexp(ValueError, "should have rank >= 2"):
                MatrixVariateNormalCholesky(
                    tf.zeros([1, 2]), tf.zeros([1, 1]), tf.zeros([1]))
            with self.assertRaisesRegexp(ValueError, 'compatible'):
                MatrixVariateNormalCholesky(
                    tf.zeros([1, 2, 3]),
                    tf.placeholder(tf.float32, [1, 3, 3]),
                    tf.placeholder(tf.float32, [1, 3, 3]))
            with self.assertRaisesRegexp(ValueError, 'compatible'):
                MatrixVariateNormalCholesky(
                    tf.zeros([1, 2, 3]),
                    tf.placeholder(tf.float32, [1, 2, 2]),
                    tf.placeholder(tf.float32, [1, 2, 2]))
            with self.assertRaisesRegexp(ValueError, 'compatible'):
                MatrixVariateNormalCholesky(
                    tf.zeros([2, 3]),
                    tf.placeholder(tf.float32, [1, 2, 2]),
                    tf.placeholder(tf.float32, [1, 3, 3]))
            u = tf.placeholder(tf.float32, [None])
            len_u = tf.shape(u)[0]
            v = tf.placeholder(tf.float32, [None])
            len_v = tf.shape(v)[0]
            dst = MatrixVariateNormalCholesky(
                tf.zeros([2, 3]), tf.zeros([len_u, len_u]),
                tf.zeros([len_v, len_v]))
            with self.assertRaisesRegexp(
                    tf.errors.InvalidArgumentError, 'compatible'):
                dst.sample().eval(
                    feed_dict={u: np.ones((3,)), v: np.ones((3,))})
                dst.sample().eval(
                    feed_dict={u: np.ones((2,)), v: np.ones((2,))})

    def test_shape_inference(self):
        with self.session(use_gpu=True):
            # Static
            mean = 10 * np.random.normal(size=(10, 11, 2, 3)).astype('d')
            u_tril = np.zeros((10, 11, 2, 2))
            v_tril = np.zeros((10, 11, 3, 3))
            dst = MatrixVariateNormalCholesky(
                tf.constant(mean), tf.constant(u_tril), tf.constant(v_tril))
            self.assertEqual(dst.get_batch_shape().as_list(), [10, 11])
            self.assertEqual(dst.get_value_shape().as_list(), [2, 3])
            # Dynamic
            unk_mean = tf.placeholder(tf.float32, None)
            unk_u_tril = tf.placeholder(tf.float32, None)
            unk_v_tril = tf.placeholder(tf.float32, None)
            dst = MatrixVariateNormalCholesky(unk_mean, unk_u_tril, unk_v_tril)
            self.assertEqual(dst.get_value_shape().as_list(), [None, None])
            feed_dict = {unk_mean: np.ones((2, 3)), unk_u_tril: np.eye(2),
                         unk_v_tril: np.eye(3)}
            self.assertEqual(list(dst.batch_shape.eval(feed_dict)), [])
            self.assertEqual(list(dst.value_shape.eval(feed_dict)), [2, 3])

    def _gen_test_params(self, seed):
        np.random.seed(seed)
        mean = 10 * np.random.normal(size=(10, 11, 2, 3)).astype('d')
        u = np.zeros((10, 11, 2, 2))
        v = np.zeros((10, 11, 3, 3))
        u_chol = np.zeros_like(u)
        v_chol = np.zeros_like(v)
        for i in range(10):
            for j in range(11):
                u[i, j] = stats.invwishart.rvs(2, np.eye(2))
                u[i, j] /= np.max(np.diag(u[i, j]))
                u_chol[i, j, :, :] = np.linalg.cholesky(u[i, j])

                v[i, j] = stats.invwishart.rvs(3, np.eye(3))
                v[i, j] /= np.max(np.diag(v[i, j]))
                v_chol[i, j, :, :] = np.linalg.cholesky(v[i, j])
        return mean, u, u_chol, v, v_chol

    @contextmanager
    def fixed_randomness_session(self, seed):
        with tf.Graph().as_default() as g:
            with self.session(use_gpu=True, graph=g):
                tf.set_random_seed(seed)
                yield

    def test_sample(self):
        with self.fixed_randomness_session(233):
            def test_sample_with(seed):
                mean, u, u_chol, v, v_chol = self._gen_test_params(seed)
                dst = MatrixVariateNormalCholesky(
                    tf.constant(mean), tf.constant(u_chol),
                    tf.constant(v_chol))
                n_exp = 20000
                samples = dst.sample(n_exp)
                sample_shape = (n_exp, 10, 11, 2, 3)
                self.assertEqual(samples.shape.as_list(), list(sample_shape))
                samples = dst.sample(n_exp).eval()
                self.assertEqual(samples.shape, sample_shape)
                self.assertAllClose(
                    np.mean(samples, axis=0), mean, rtol=5e-2, atol=5e-2)
                samples = np.reshape(samples.transpose([0, 1, 2, 4, 3]),
                                     [n_exp, 10, 11, -1])
                for i in range(10):
                    for j in range(11):
                        for k in range(3):
                            self.assertAllClose(np.cov(samples[:, i, j, :].T),
                                                np.kron(v[i, j], u[i, j]),
                                                rtol=1e-1, atol=1e-1)

            for seed in [23, 233, 2333]:
                test_sample_with(seed)

    def test_prob(self):
        with self.fixed_randomness_session(233):
            def test_prob_with(seed):
                mean, u, u_chol, v, v_chol = self._gen_test_params(seed)
                dst = MatrixVariateNormalCholesky(
                    tf.constant(mean), tf.constant(u_chol),
                    tf.constant(v_chol), check_numerics=True)
                n_exp = 200
                samples = dst.sample(n_exp).eval()
                log_pdf = dst.log_prob(tf.constant(samples))
                pdf_shape = (n_exp, 10, 11)
                self.assertEqual(log_pdf.shape.as_list(), list(pdf_shape))
                log_pdf = log_pdf.eval()
                self.assertEqual(log_pdf.shape, pdf_shape)
                for i in range(10):
                    for j in range(11):
                        log_pdf_exact = stats.matrix_normal.logpdf(
                            samples[:, i, j, :], mean[i, j],
                            u[i, j], v[i, j])
                        self.assertAllClose(
                            log_pdf_exact, log_pdf[:, i, j])
                self.assertAllClose(
                    np.exp(log_pdf), dst.prob(tf.constant(samples)).eval())

            for seed in [23, 233, 2333]:
                test_prob_with(seed)

    def test_sample_reparameterized(self):
        mean, u, u_chol, v, v_chol = self._gen_test_params(23)
        mean = tf.constant(mean)
        u_chol = tf.constant(u_chol)
        v_chol = tf.constant(v_chol)
        mvn_rep = MatrixVariateNormalCholesky(mean, u_chol, v_chol)
        samples = mvn_rep.sample(tf.placeholder(tf.int32, shape=[]))
        mean_grads, u_grads, v_grads = tf.gradients(
            samples, [mean, u_chol, v_chol])
        self.assertTrue(mean_grads is not None)
        self.assertTrue(u_grads is not None)
        self.assertTrue(v_grads is not None)
        mvn_rep = MatrixVariateNormalCholesky(mean, u_chol, v_chol,
                                              is_reparameterized=False)
        samples = mvn_rep.sample(tf.placeholder(tf.int32, shape=[]))
        mean_grads, u_grads, v_grads = tf.gradients(
            samples, [mean, u_chol, v_chol])
        self.assertEqual(mean_grads, None)
        self.assertEqual(u_grads, None)
        self.assertEqual(v_grads, None)
