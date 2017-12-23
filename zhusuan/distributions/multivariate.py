#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import numpy as np
import tensorflow as tf

from zhusuan.distributions.base import Distribution
from zhusuan.distributions.utils import \
        maybe_explicit_broadcast, \
        assert_same_float_dtype, \
        assert_same_float_and_int_dtype, \
        assert_rank_at_least, \
        assert_rank_at_least_one, \
        assert_scalar, \
        assert_positive_int32_scalar, \
        get_shape_at, \
        get_shape_list, \
        open_interval_standard_uniform, \
        log_combination


__all__ = [
    'MultivariateNormalCholesky',
    'Multinomial',
    'UnnormalizedMultinomial',
    'OnehotCategorical',
    'OnehotDiscrete',
    'Dirichlet',
    'ExpConcrete',
    'ExpGumbelSoftmax',
    'Concrete',
    'GumbelSoftmax',
]


class MultivariateNormalCholesky(Distribution):
    """
    The class of multivariate normal distribution, where covariance is
    parameterized with the lower triangular matrix in Cholesky decomposition,

        .. math :: L \\text{s.t.} LL^T = \Sigma.

    See :class:`~zhusuan.distributions.base.Distribution` for details.

    :param mean: An N-D `float` Tensor of shape [..., n_dim]. Each slice
        [i, j, ..., k, :] represents the mean of a single multivariate normal
        distribution.
    :param cov_tril: An (N+1)-D `float` Tensor of shape [..., n_dim, n_dim].
        Each slice [i, ..., k, :, :] represents the lower triangular matrix in
        the Cholesky decomposition of the covariance of a single distribution.
    :param group_ndims: A 0-D `int32` Tensor representing the number of
        dimensions in `batch_shape` (counted from the end) that are grouped
        into a single event, so that their probabilities are calculated
        together. Default is 0, which means a single value is an event.
        See :class:`~zhusuan.distributions.base.Distribution` for more detailed
        explanation.
    :param is_reparameterized: A Bool. If True, gradients on samples from this
        distribution are allowed to propagate into inputs, using the
        reparametrization trick from (Kingma, 2013).
    :param use_path_derivative: A bool. Whether when taking the gradients
        of the log-probability to propagate them through the parameters
        of the distribution (False meaning you do propagate them). This
        is based on the paper "Sticking the Landing: Simple,
        Lower-Variance Gradient Estimators for Variational Inference"
    :param check_numerics: Bool. Whether to check numeric issues.
    """

    def __init__(self,
                 mean,
                 cov_tril,
                 group_ndims=0,
                 is_reparameterized=True,
                 use_path_derivative=False,
                 check_numerics=False,
                 **kwargs):
        self._check_numerics = check_numerics
        self._mean = tf.convert_to_tensor(mean)
        self._mean = assert_rank_at_least_one(
            self._mean, 'MultivariateNormalCholesky.mean')
        self._n_dim = get_shape_at(self._mean, -1)
        self._cov_tril = tf.convert_to_tensor(cov_tril)
        self._cov_tril = assert_rank_at_least(
            self._cov_tril, 2, 'MultivariateNormalCholesky.cov_tril')

        # Static shape check
        expected_shape = self._mean.get_shape().concatenate(
            [self._n_dim if isinstance(self._n_dim, int) else None])
        self._cov_tril.get_shape().assert_is_compatible_with(expected_shape)
        # Dynamic
        expected_shape = tf.concat(
            [tf.shape(self._mean), [self._n_dim]], axis=0)
        actual_shape = tf.shape(self._cov_tril)
        msg = ['MultivariateNormalCholesky.cov_tril should have compatible '
               'shape with mean. Expected', expected_shape, ' got ',
               actual_shape]
        assert_ops = [tf.assert_equal(expected_shape, actual_shape, msg)]
        with tf.control_dependencies(assert_ops):
            self._cov_tril = tf.identity(self._cov_tril)

        dtype = assert_same_float_dtype(
            [(self._mean, 'MultivariateNormalCholesky.mean'),
             (self._cov_tril, 'MultivariateNormalCholesky.cov_tril')])
        super(MultivariateNormalCholesky, self).__init__(
            dtype=dtype,
            param_dtype=dtype,
            is_continuous=True,
            is_reparameterized=is_reparameterized,
            use_path_derivative=use_path_derivative,
            group_ndims=group_ndims,
            **kwargs)

    @property
    def mean(self):
        """The mean of the normal distribution."""
        return self._mean

    @property
    def cov_tril(self):
        """
        The lower triangular matrix in the cholosky decomposition of the
        covariance.
        """
        return self._cov_tril

    def _value_shape(self):
        return tf.convert_to_tensor([self._n_dim], tf.int32)

    def _get_value_shape(self):
        if isinstance(self._n_dim, int):
            return tf.TensorShape([self._n_dim])
        return tf.TensorShape([None])

    def _batch_shape(self):
        return tf.shape(self.mean)[:-1]

    def _get_batch_shape(self):
        if self.mean.get_shape():
            return self.mean.get_shape()[:-1]
        return tf.TensorShape(None)

    def _sample(self, n_samples):
        mean, cov_tril = self.mean, self.cov_tril
        if not self.is_reparameterized:
            mean = tf.stop_gradient(mean)
            cov_tril = tf.stop_gradient(cov_tril)

        def tile(t):
            new_shape = tf.concat([[n_samples], tf.ones_like(tf.shape(t))], 0)
            return tf.tile(tf.expand_dims(t, 0), new_shape)

        batch_mean = tile(mean)
        batch_cov = tile(cov_tril)
        # n_dim -> n_dim x 1 for matmul
        batch_mean = tf.expand_dims(batch_mean, -1)
        noise = tf.random_normal(tf.shape(batch_mean), dtype=self.dtype)
        samples = tf.matmul(batch_cov, noise) + batch_mean
        samples = tf.squeeze(samples, -1)
        # Update static shape
        static_n_samples = n_samples if isinstance(n_samples, int) else None
        samples.set_shape(tf.TensorShape([static_n_samples])
                          .concatenate(self.get_batch_shape())
                          .concatenate(self.get_value_shape()))
        return samples

    def _log_prob(self, given):
        mean, cov_tril = (self.path_param(self.mean),
                          self.path_param(self.cov_tril))
        log_det = 2 * tf.reduce_sum(
            tf.log(tf.matrix_diag_part(cov_tril)), axis=-1)
        N = tf.cast(self._n_dim, self.dtype)
        logZ = - N / 2 * tf.log(2 * tf.constant(np.pi, dtype=self.dtype)) - \
            log_det / 2
        # logZ.shape == batch_shape
        if self._check_numerics:
            logZ = tf.check_numerics(logZ, "log[det(Cov)]")
        # (given-mean)' Sigma^{-1} (given-mean) =
        # (g-m)' L^{-T} L^{-1} (g-m) = |x|^2, where Lx = g-m =: y.
        y = tf.expand_dims(given - mean, -1)
        L, _ = maybe_explicit_broadcast(
            cov_tril, y, 'MultivariateNormalCholesky.cov_tril',
            'expand_dims(given, -1)')
        x = tf.matrix_triangular_solve(L, y, lower=True)
        x = tf.squeeze(x, -1)
        stoc_dist = -0.5 * tf.reduce_sum(tf.square(x), axis=-1)
        return logZ + stoc_dist

    def _prob(self, given):
        return tf.exp(self._log_prob(given))


class Multinomial(Distribution):
    """
    The class of Multinomial distribution.
    See :class:`~zhusuan.distributions.base.Distribution` for details.

    :param logits: A N-D (N >= 1) `float` Tensor of shape (...,
        n_categories). Each slice `[i, j, ..., k, :]` represents the
        un-normalized log probabilities for all categories.

        .. math:: \\mathrm{logits} \\propto \\log p

    :param n_experiments: A 0-D `int32` Tensor or `None`. When it is a 0-D
        `int32` integer, it represents the number of experiments for each
        sample, which should be invariant among samples. In this situation
        `_sample` function is supported. When it is `None`, `_sample` function
        is not supported, and when calculating probabilities the number of
        experiments will be inferred from `given`, so it could vary among
        samples.
    :param normalize_logits: A bool indicating whether `logits` should be
        normalized when computing probability. If you believe `logits` is
        already normalized, set it to `False` to speed up. Default is True.
    :param dtype: The value type of samples from the distribution.
    :param group_ndims: A 0-D `int32` Tensor representing the number of
        dimensions in `batch_shape` (counted from the end) that are grouped
        into a single event, so that their probabilities are calculated
        together. Default is 0, which means a single value is an event.
        See :class:`~zhusuan.distributions.base.Distribution` for more detailed
        explanation.

    A single sample is a N-D Tensor with the same shape as logits. Each slice
    `[i, j, ..., k, :]` is a vector of counts for all categories.
    """

    def __init__(self,
                 logits,
                 n_experiments,
                 normalize_logits=True,
                 dtype=None,
                 group_ndims=0,
                 **kwargs):
        self._logits = tf.convert_to_tensor(logits)
        param_dtype = assert_same_float_dtype(
            [(self._logits, 'Multinomial.logits')])

        if dtype is None:
            dtype = tf.int32
        assert_same_float_and_int_dtype([], dtype)

        self._logits = assert_rank_at_least_one(
            self._logits, 'Multinomial.logits')
        self._n_categories = get_shape_at(self._logits, -1)

        if n_experiments is None:
            self._n_experiments = None
        else:
            self._n_experiments = assert_positive_int32_scalar(
                n_experiments, 'Multinomial.n_experiments')

        self.normalize_logits = normalize_logits

        super(Multinomial, self).__init__(
            dtype=dtype,
            param_dtype=param_dtype,
            is_continuous=False,
            is_reparameterized=False,
            group_ndims=group_ndims,
            **kwargs)

    @property
    def logits(self):
        """The un-normalized log probabilities."""
        return self._logits

    @property
    def n_categories(self):
        """The number of categories in the distribution."""
        return self._n_categories

    @property
    def n_experiments(self):
        """The number of experiments for each sample."""
        return self._n_experiments

    def _value_shape(self):
        return tf.convert_to_tensor([self.n_categories], tf.int32)

    def _get_value_shape(self):
        if isinstance(self.n_categories, int):
            return tf.TensorShape([self.n_categories])
        return tf.TensorShape([None])

    def _batch_shape(self):
        return tf.shape(self.logits)[:-1]

    def _get_batch_shape(self):
        if self.logits.get_shape():
            return self.logits.get_shape()[:-1]
        return tf.TensorShape(None)

    def _sample(self, n_samples):
        if self.n_experiments is None:
            raise ValueError('Cannot sample when `n_experiments` is None')

        if self.logits.get_shape().ndims == 2:
            logits_flat = self.logits
        else:
            logits_flat = tf.reshape(self.logits, [-1, self.n_categories])
        samples_flat = tf.transpose(
            tf.multinomial(logits_flat, n_samples * self.n_experiments))
        shape = tf.concat([[n_samples, self.n_experiments],
                           self.batch_shape], 0)
        samples = tf.reshape(samples_flat, shape)
        static_n_samples = n_samples if isinstance(n_samples,
                                                   int) else None
        static_n_exps = self.n_experiments \
            if isinstance(self.n_experiments, int) else None
        samples.set_shape(
            tf.TensorShape([static_n_samples, static_n_exps]).
            concatenate(self.get_batch_shape()))
        samples = tf.reduce_sum(
            tf.one_hot(samples, self.n_categories, dtype=self.dtype),
            axis=1)
        return samples

    def _log_prob(self, given):
        given = tf.cast(given, self.param_dtype)
        given, logits = maybe_explicit_broadcast(
            given, self.logits, 'given', 'logits')
        if self.normalize_logits:
            logits = logits - tf.reduce_logsumexp(
                logits, axis=-1, keep_dims=True)
        if self.n_experiments is None:
            n = tf.reduce_sum(given, -1)
        else:
            n = tf.cast(self.n_experiments, self.param_dtype)
        log_p = log_combination(n, given) + \
            tf.reduce_sum(given * logits, -1)
        return log_p

    def _prob(self, given):
        return tf.exp(self._log_prob(given))


class UnnormalizedMultinomial(Distribution):
    """
    The class of UnnormalizedMultinomial distribution.
    UnnormalizedMultinomial distribution calculates probabilities differently
    from :class:`Multinomial`: It considers the bag-of-words `given` as a
    statistics of an ordered result sequence, and calculates the probability
    of the (imagined) ordered sequence. Hence it does not multiply the term

    .. math::

        \\binom{n}{k_1, k_2, \\dots} =  \\frac{n!}{\\prod_{i} k_i!}

    See :class:`~zhusuan.distributions.base.Distribution` for details.

    :param logits: A N-D (N >= 1) `float` Tensor of shape (...,
        n_categories). Each slice `[i, j, ..., k, :]` represents the
        un-normalized log probabilities for all categories.

        .. math:: \\mathrm{logits} \\propto \\log p

    :param normalize_logits: A bool indicating whether `logits` should be
        normalized when computing probability. If you believe `logits` is
        already normalized, set it to `False` to speed up. Default is True.
    :param dtype: The value type of samples from the distribution.
    :param group_ndims: A 0-D `int32` Tensor representing the number of
        dimensions in `batch_shape` (counted from the end) that are grouped
        into a single event, so that their probabilities are calculated
        together. Default is 0, which means a single value is an event.
        See :class:`~zhusuan.distributions.base.Distribution` for more detailed
        explanation.

    A single sample is a N-D Tensor with the same shape as logits. Each slice
    `[i, j, ..., k, :]` is a vector of counts for all categories.
    """

    def __init__(self,
                 logits,
                 normalize_logits=True,
                 dtype=None,
                 group_ndims=0,
                 **kwargs):
        self._logits = tf.convert_to_tensor(logits)
        param_dtype = assert_same_float_dtype(
            [(self._logits, 'UnnormalizedMultinomial.logits')])

        if dtype is None:
            dtype = tf.int32
        assert_same_float_and_int_dtype([], dtype)

        self._logits = assert_rank_at_least_one(
            self._logits, 'UnnormalizedMultinomial.logits')
        self._n_categories = get_shape_at(self._logits, -1)

        self.normalize_logits = normalize_logits

        super(UnnormalizedMultinomial, self).__init__(
            dtype=dtype,
            param_dtype=param_dtype,
            is_continuous=False,
            is_reparameterized=False,
            group_ndims=group_ndims,
            **kwargs)

    @property
    def logits(self):
        """The un-normalized log probabilities."""
        return self._logits

    @property
    def n_categories(self):
        """The number of categories in the distribution."""
        return self._n_categories

    def _value_shape(self):
        return tf.convert_to_tensor([self.n_categories], tf.int32)

    def _get_value_shape(self):
        if isinstance(self.n_categories, int):
            return tf.TensorShape([self.n_categories])
        return tf.TensorShape([None])

    def _batch_shape(self):
        return tf.shape(self.logits)[:-1]

    def _get_batch_shape(self):
        if self.logits.get_shape():
            return self.logits.get_shape()[:-1]
        return tf.TensorShape(None)

    def _sample(self, n_samples):
        raise NotImplementedError("Unnormalized multinomial distribution"
                                  " does not support sampling because"
                                  " n_experiments is not given. Please use"
                                  " class Multinomial to sample")

    def _log_prob(self, given):
        given = tf.cast(given, self.param_dtype)
        given, logits = maybe_explicit_broadcast(
            given, self.logits, 'given', 'logits')
        if self.normalize_logits:
            logits = logits - tf.reduce_logsumexp(
                logits, axis=-1, keep_dims=True)
        log_p = tf.reduce_sum(given * logits, -1)
        return log_p

    def _prob(self, given):
        return tf.exp(self._log_prob(given))


BagofCategorical = UnnormalizedMultinomial


class OnehotCategorical(Distribution):
    """
    The class of one-hot Categorical distribution.
    See :class:`~zhusuan.distributions.base.Distribution` for details.

    :param logits: A N-D (N >= 1) `float` Tensor of shape (...,
        n_categories). Each slice `[i, j, ..., k, :]` represents the
        un-normalized log probabilities for all categories.

        .. math:: \\mathrm{logits} \\propto \\log p

    :param dtype: The value type of samples from the distribution.
    :param group_ndims: A 0-D `int32` Tensor representing the number of
        dimensions in `batch_shape` (counted from the end) that are grouped
        into a single event, so that their probabilities are calculated
        together. Default is 0, which means a single value is an event.
        See :class:`~zhusuan.distributions.base.Distribution` for more detailed
        explanation.

    A single sample is a N-D Tensor with the same shape as logits. Each slice
    `[i, j, ..., k, :]` is a one-hot vector of the selected category.
    """

    def __init__(self, logits, dtype=None, group_ndims=0, **kwargs):
        self._logits = tf.convert_to_tensor(logits)
        param_dtype = assert_same_float_dtype(
            [(self._logits, 'OnehotCategorical.logits')])

        if dtype is None:
            dtype = tf.int32
        assert_same_float_and_int_dtype([], dtype)

        self._logits = assert_rank_at_least_one(
            self._logits, 'OnehotCategorical.logits')
        self._n_categories = get_shape_at(self._logits, -1)

        super(OnehotCategorical, self).__init__(
            dtype=dtype,
            param_dtype=param_dtype,
            is_continuous=False,
            is_reparameterized=False,
            group_ndims=group_ndims,
            **kwargs)

    @property
    def logits(self):
        """The un-normalized log probabilities."""
        return self._logits

    @property
    def n_categories(self):
        """The number of categories in the distribution."""
        return self._n_categories

    def _value_shape(self):
        return tf.convert_to_tensor([self.n_categories], tf.int32)

    def _get_value_shape(self):
        if isinstance(self.n_categories, int):
            return tf.TensorShape([self.n_categories])
        return tf.TensorShape([None])

    def _batch_shape(self):
        return tf.shape(self.logits)[:-1]

    def _get_batch_shape(self):
        if self.logits.get_shape():
            return self.logits.get_shape()[:-1]
        return tf.TensorShape(None)

    def _sample(self, n_samples):
        if self.logits.get_shape().ndims == 2:
            logits_flat = self.logits
        else:
            logits_flat = tf.reshape(self.logits, [-1, self.n_categories])
        samples_flat = tf.transpose(tf.multinomial(logits_flat, n_samples))
        if self.logits.get_shape().ndims == 2:
            samples = samples_flat
        else:
            shape = tf.concat([[n_samples], self.batch_shape], 0)
            samples = tf.reshape(samples_flat, shape)
            static_n_samples = n_samples if isinstance(n_samples,
                                                       int) else None
            samples.set_shape(
                tf.TensorShape([static_n_samples]).
                concatenate(self.get_batch_shape()))
        samples = tf.one_hot(samples, self.n_categories, dtype=self.dtype)
        return samples

    def _log_prob(self, given):
        given = tf.cast(given, self.param_dtype)
        given, logits = maybe_explicit_broadcast(
            given, self.logits, 'given', 'logits')
        if (given.get_shape().ndims == 2) or (logits.get_shape().ndims == 2):
            given_flat = given
            logits_flat = logits
        else:
            given_flat = tf.reshape(given, [-1, self.n_categories])
            logits_flat = tf.reshape(logits, [-1, self.n_categories])
        log_p_flat = -tf.nn.softmax_cross_entropy_with_logits(
            labels=given_flat, logits=logits_flat)
        if (given.get_shape().ndims == 2) or (logits.get_shape().ndims == 2):
            log_p = log_p_flat
        else:
            log_p = tf.reshape(log_p_flat, tf.shape(logits)[:-1])
            if given.get_shape() and logits.get_shape():
                log_p.set_shape(tf.broadcast_static_shape(
                    given.get_shape(), logits.get_shape())[:-1])
        return log_p

    def _prob(self, given):
        return tf.exp(self._log_prob(given))


OnehotDiscrete = OnehotCategorical


class Dirichlet(Distribution):
    """
    The class of Dirichlet distribution.
    See :class:`~zhusuan.distributions.base.Distribution` for details.

    :param alpha: A N-D (N >= 1) `float` Tensor of shape (..., n_categories).
        Each slice `[i, j, ..., k, :]` represents the concentration parameter
        of a Dirichlet distribution. Should be positive.
    :param group_ndims: A 0-D `int32` Tensor representing the number of
        dimensions in `batch_shape` (counted from the end) that are grouped
        into a single event, so that their probabilities are calculated
        together. Default is 0, which means a single value is an event.
        See :class:`~zhusuan.distributions.base.Distribution` for more detailed
        explanation.

    A single sample is a N-D Tensor with the same shape as alpha. Each slice
    `[i, j, ..., k, :]` of the sample is a vector of probabilities of a
    Categorical distribution `[x_1, x_2, ... ]`, which lies on the simplex

    .. math:: \\sum_{i} x_i = 1, 0 < x_i < 1

    """

    def __init__(self,
                 alpha,
                 group_ndims=0,
                 check_numerics=False,
                 **kwargs):
        self._alpha = tf.convert_to_tensor(alpha)
        dtype = assert_same_float_dtype(
            [(self._alpha, 'Dirichlet.alpha')])

        static_alpha_shape = self._alpha.get_shape()
        shape_err_msg = "alpha should have rank >= 1."
        cat_err_msg = "n_categories (length of the last axis " \
                      "of alpha) should be at least 2."
        if static_alpha_shape and (static_alpha_shape.ndims < 1):
            raise ValueError(shape_err_msg)
        elif static_alpha_shape and (
                static_alpha_shape[-1].value is not None):
            self._n_categories = static_alpha_shape[-1].value
            if self._n_categories < 2:
                raise ValueError(cat_err_msg)
        else:
            _assert_shape_op = tf.assert_rank_at_least(
                self._alpha, 1, message=shape_err_msg)
            with tf.control_dependencies([_assert_shape_op]):
                self._alpha = tf.identity(self._alpha)
            self._n_categories = tf.shape(self._alpha)[-1]

            _assert_cat_op = tf.assert_greater_equal(
                self._n_categories, 2, message=cat_err_msg)
            with tf.control_dependencies([_assert_cat_op]):
                self._alpha = tf.identity(self._alpha)
        self._check_numerics = check_numerics

        super(Dirichlet, self).__init__(
            dtype=dtype,
            param_dtype=dtype,
            is_continuous=True,
            is_reparameterized=False,
            group_ndims=group_ndims,
            **kwargs)

    @property
    def alpha(self):
        """The concentration parameter of the Dirichlet distribution."""
        return self._alpha

    @property
    def n_categories(self):
        """The number of categories in the distribution."""
        return self._n_categories

    def _value_shape(self):
        return tf.convert_to_tensor([self.n_categories], tf.int32)

    def _get_value_shape(self):
        if isinstance(self.n_categories, int):
            return tf.TensorShape([self.n_categories])
        return tf.TensorShape([None])

    def _batch_shape(self):
        return tf.shape(self.alpha)[:-1]

    def _get_batch_shape(self):
        if self.alpha.get_shape():
            return self.alpha.get_shape()[:-1]
        return tf.TensorShape(None)

    def _sample(self, n_samples):
        samples = tf.random_gamma([n_samples], self.alpha,
                                  beta=1, dtype=self.dtype)
        return samples / tf.reduce_sum(samples, -1, keep_dims=True)

    def _log_prob(self, given):
        given, alpha = maybe_explicit_broadcast(
            given, self.alpha, 'given', 'alpha')
        lbeta_alpha = tf.lbeta(alpha)
        # fix of no static shape inference for tf.lbeta
        if alpha.get_shape():
            lbeta_alpha.set_shape(alpha.get_shape()[:-1])
        log_given = tf.log(given)
        if self._check_numerics:
            lbeta_alpha = tf.check_numerics(lbeta_alpha, "lbeta(alpha)")
            log_given = tf.check_numerics(log_given, "log(given)")
        log_p = -lbeta_alpha + tf.reduce_sum((alpha - 1) * log_given, -1)
        return log_p

    def _prob(self, given):
        return tf.exp(self._log_prob(given))


class ExpConcrete(Distribution):
    """
    The class of ExpConcrete distribution from (Maddison, 2016), transformed
    from :class:`~Concrete` by taking logarithm.
    See :class:`~zhusuan.distributions.base.Distribution` for details.

    .. seealso::

        :class:`~zhusuan.distributions.univariate.BinConcrete` and
        :class:`~Concrete`

    :param temperature: A 0-D `float` Tensor. The temperature of the relaxed
        distribution. The temperature should be positive.
    :param logits: A N-D (N >= 1) `float` Tensor of shape (...,
        n_categories). Each slice `[i, j, ..., k, :]` represents the
        un-normalized log probabilities for all categories.

        .. math:: \\mathrm{logits} \\propto \\log p

    :param group_ndims: A 0-D `int32` Tensor representing the number of
        dimensions in `batch_shape` (counted from the end) that are grouped
        into a single event, so that their probabilities are calculated
        together. Default is 0, which means a single value is an event.
        See :class:`~zhusuan.distributions.base.Distribution` for more detailed
        explanation.
    :param is_reparameterized: A Bool. If True, gradients on samples from this
        distribution are allowed to propagate into inputs, using the
        reparametrization trick from (Kingma, 2013).
    :param use_path_derivative: A bool. Whether when taking the gradients
        of the log-probability to propagate them through the parameters
        of the distribution (False meaning you do propagate them). This
        is based on the paper "Sticking the Landing: Simple,
        Lower-Variance Gradient Estimators for Variational Inference"
    :param check_numerics: Bool. Whether to check numeric issues.
    """

    def __init__(self,
                 temperature,
                 logits,
                 group_ndims=0,
                 is_reparameterized=True,
                 use_path_derivative=False,
                 check_numerics=False,
                 **kwargs):
        self._logits = tf.convert_to_tensor(logits)
        self._temperature = tf.convert_to_tensor(temperature)
        param_dtype = assert_same_float_dtype(
            [(self._logits, 'ExpConcrete.logits'),
             (self._temperature, 'ExpConcrete.temperature')])

        self._logits = assert_rank_at_least_one(
            self._logits, 'ExpConcrete.logits')
        self._n_categories = get_shape_at(self._logits, -1)

        self._temperature = assert_scalar(
            self._temperature, 'ExpConcrete.temperature')

        self._check_numerics = check_numerics
        super(ExpConcrete, self).__init__(
            dtype=param_dtype,
            param_dtype=param_dtype,
            is_continuous=True,
            is_reparameterized=is_reparameterized,
            use_path_derivative=use_path_derivative,
            group_ndims=group_ndims,
            **kwargs)

    @property
    def temperature(self):
        """The temperature of ExpConcrete."""
        return self._temperature

    @property
    def logits(self):
        """The un-normalized log probabilities."""
        return self._logits

    @property
    def n_categories(self):
        """The number of categories in the distribution."""
        return self._n_categories

    def _value_shape(self):
        return tf.convert_to_tensor([self.n_categories], tf.int32)

    def _get_value_shape(self):
        if isinstance(self.n_categories, int):
            return tf.TensorShape([self.n_categories])
        return tf.TensorShape([None])

    def _batch_shape(self):
        return tf.shape(self.logits)[:-1]

    def _get_batch_shape(self):
        if self.logits.get_shape():
            return self.logits.get_shape()[:-1]
        return tf.TensorShape(None)

    def _sample(self, n_samples):
        logits, temperature = self.logits, self.temperature
        if not self.is_reparameterized:
            logits = tf.stop_gradient(logits)
            temperature = tf.stop_gradient(temperature)
        shape = tf.concat([[n_samples], tf.shape(self.logits)], 0)

        uniform = open_interval_standard_uniform(shape, self.dtype)
        gumbel = -tf.log(-tf.log(uniform))
        samples = tf.nn.log_softmax((logits + gumbel) / temperature)

        static_n_samples = n_samples if isinstance(n_samples, int) else None
        samples.set_shape(
            tf.TensorShape([static_n_samples]).concatenate(logits.get_shape()))
        return samples

    def _log_prob(self, given):
        logits, temperature = self.path_param(self.logits),\
                              self.path_param(self.temperature)
        n = tf.cast(self.n_categories, self.dtype)
        log_temperature = tf.log(temperature)

        if self._check_numerics:
            log_temperature = tf.check_numerics(
                log_temperature, "log(temperature)")

        temp = logits - temperature * given

        return tf.lgamma(n) + (n - 1) * log_temperature + \
            tf.reduce_sum(temp, axis=-1) - \
            n * tf.reduce_logsumexp(temp, axis=-1)

    def _prob(self, given):
        return tf.exp(self._log_prob(given))


ExpGumbelSoftmax = ExpConcrete


class Concrete(Distribution):
    """
    The class of Concrete (or Gumbel-Softmax) distribution from
    (Maddison, 2016; Jang, 2016), served as the
    continuous relaxation of the :class:`~OnehotCategorical`.
    See :class:`~zhusuan.distributions.base.Distribution` for details.

    .. seealso::

        :class:`~zhusuan.distributions.univariate.BinConcrete` and
        :class:`~ExpConcrete`

    :param temperature: A 0-D `float` Tensor. The temperature of the relaxed
        distribution. The temperature should be positive.
    :param logits: A N-D (N >= 1) `float` Tensor of shape (...,
        n_categories). Each slice `[i, j, ..., k, :]` represents the
        un-normalized log probabilities for all categories.

        .. math:: \\mathrm{logits} \\propto \\log p

    :param group_ndims: A 0-D `int32` Tensor representing the number of
        dimensions in `batch_shape` (counted from the end) that are grouped
        into a single event, so that their probabilities are calculated
        together. Default is 0, which means a single value is an event.
        See :class:`~zhusuan.distributions.base.Distribution` for more detailed
        explanation.
    :param is_reparameterized: A Bool. If True, gradients on samples from this
        distribution are allowed to propagate into inputs, using the
        reparametrization trick from (Kingma, 2013).
    :param use_path_derivative: A bool. Whether when taking the gradients
        of the log-probability to propagate them through the parameters
        of the distribution (False meaning you do propagate them). This
        is based on the paper "Sticking the Landing: Simple,
        Lower-Variance Gradient Estimators for Variational Inference"
    :param check_numerics: Bool. Whether to check numeric issues.
    """

    def __init__(self,
                 temperature,
                 logits,
                 group_ndims=0,
                 is_reparameterized=True,
                 use_path_derivative=False,
                 check_numerics=False,
                 **kwargs):
        self._logits = tf.convert_to_tensor(logits)
        self._temperature = tf.convert_to_tensor(temperature)
        param_dtype = assert_same_float_dtype(
            [(self._logits, 'Concrete.logits'),
             (self._temperature, 'Concrete.temperature')])

        self._logits = assert_rank_at_least_one(
            self._logits, 'Concrete.logits')
        self._n_categories = get_shape_at(self._logits, -1)

        self._temperature = assert_scalar(
            self._temperature, 'Concrete.temperature')

        self._check_numerics = check_numerics
        super(Concrete, self).__init__(
            dtype=param_dtype,
            param_dtype=param_dtype,
            is_continuous=True,
            is_reparameterized=is_reparameterized,
            use_path_derivative=use_path_derivative,
            group_ndims=group_ndims,
            **kwargs)

    @property
    def temperature(self):
        """The temperature of Concrete."""
        return self._temperature

    @property
    def logits(self):
        """The un-normalized log probabilities."""
        return self._logits

    @property
    def n_categories(self):
        """The number of categories in the distribution."""
        return self._n_categories

    def _value_shape(self):
        return tf.convert_to_tensor([self.n_categories], tf.int32)

    def _get_value_shape(self):
        if isinstance(self.n_categories, int):
            return tf.TensorShape([self.n_categories])
        return tf.TensorShape([None])

    def _batch_shape(self):
        return tf.shape(self.logits)[:-1]

    def _get_batch_shape(self):
        if self.logits.get_shape():
            return self.logits.get_shape()[:-1]
        return tf.TensorShape(None)

    def _sample(self, n_samples):
        logits, temperature = self.logits, self.temperature
        if not self.is_reparameterized:
            logits = tf.stop_gradient(logits)
            temperature = tf.stop_gradient(temperature)
        shape = tf.concat([[n_samples], tf.shape(self.logits)], 0)

        uniform = open_interval_standard_uniform(shape, self.dtype)
        # TODO: Add Gumbel distribution
        gumbel = -tf.log(-tf.log(uniform))
        samples = tf.nn.softmax((logits + gumbel) / temperature)

        static_n_samples = n_samples if isinstance(n_samples, int) else None
        samples.set_shape(
            tf.TensorShape([static_n_samples]).concatenate(logits.get_shape()))
        return samples

    def _log_prob(self, given):
        logits, temperature = self.path_param(self.logits), \
                              self.path_param(self.temperature)
        log_given = tf.log(given)
        log_temperature = tf.log(temperature)
        n = tf.cast(self.n_categories, self.dtype)

        if self._check_numerics:
            log_given = tf.check_numerics(log_given, "log(given)")
            log_temperature = tf.check_numerics(
                log_temperature, "log(temperature)")

        temp = logits - temperature * log_given

        return tf.lgamma(n) + (n - 1) * log_temperature + \
            tf.reduce_sum(temp - log_given, axis=-1) - \
            n * tf.reduce_logsumexp(temp, axis=-1)

    def _prob(self, given):
        return tf.exp(self._log_prob(given))


GumbelSoftmax = Concrete
