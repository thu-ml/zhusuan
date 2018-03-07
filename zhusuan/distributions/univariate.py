#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import numpy as np
import tensorflow as tf

from zhusuan.distributions.base import Distribution
from zhusuan.distributions.utils import (
    maybe_explicit_broadcast,
    assert_same_dtype_in,
    assert_same_float_dtype,
    assert_dtype_in_dtypes,
    assert_dtype_is_int_or_float,
    assert_scalar,
    assert_rank_at_least_one,
    get_shape_at,
    open_interval_standard_uniform,
    ensure_logstd_std_order_change,
)


__all__ = [
    'Normal',
    'FoldNormal',
    'Bernoulli',
    'Categorical',
    'Discrete',
    'Uniform',
    'Gamma',
    'Beta',
    'Poisson',
    'Binomial',
    'InverseGamma',
    'Laplace',
    'BinConcrete',
    'BinGumbelSoftmax',
]


class Normal(Distribution):
    """
    The class of univariate Normal distribution.
    See :class:`~zhusuan.distributions.base.Distribution` for details.

    .. warning::

        The order of arguments `logstd`/`std` has changed to `std`/`logstd`
        since 0.3.1. Please use named arguments: ``Normal(mean, std=..., ...)``
        or ``Normal(mean, logstd=..., ...)``.

    :param mean: A `float` Tensor. The mean of the Normal distribution.
        Should be broadcastable to match `std` or `logstd`.
    :param _sentinel: Used to prevent positional parameters. Internal,
        do not use.
    :param std: A `float` Tensor. The standard deviation of the Normal
        distribution. Should be positive and broadcastable to match `mean`.
    :param logstd: A `float` Tensor. The log standard deviation of the Normal
        distribution. Should be broadcastable to match `mean`.
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
                 mean=0.,
                 _sentinel=None,
                 std=None,
                 logstd=None,
                 group_ndims=0,
                 is_reparameterized=True,
                 use_path_derivative=False,
                 check_numerics=False,
                 **kwargs):
        ensure_logstd_std_order_change("Normal", _sentinel)
        self._mean = tf.convert_to_tensor(mean)

        if (logstd is None) == (std is None):
            raise ValueError("Either std or logstd should be passed but not "
                             "both of them.")
        elif logstd is None:
            self._std = tf.convert_to_tensor(std)
            dtype = assert_same_float_dtype([(self._mean, 'Normal.mean'),
                                             (self._std, 'Normal.std')])
            logstd = tf.log(self._std)
            if check_numerics:
                logstd = tf.check_numerics(logstd, "log(std)")
            self._logstd = logstd
        else:
            # std is None
            self._logstd = tf.convert_to_tensor(logstd)
            dtype = assert_same_float_dtype([(self._mean, 'Normal.mean'),
                                             (self._logstd, 'Normal.logstd')])
            std = tf.exp(self._logstd)
            if check_numerics:
                std = tf.check_numerics(std, "exp(logstd)")
            self._std = std

        try:
            tf.broadcast_static_shape(self._mean.get_shape(),
                                      self._std.get_shape())
        except ValueError:
            raise ValueError(
                "mean and std/logstd should be broadcastable to match each "
                "other. ({} vs. {})".format(
                    self._mean.get_shape(), self._std.get_shape()))
        self._check_numerics = check_numerics
        super(Normal, self).__init__(
            dtype=dtype,
            param_dtype=dtype,
            is_continuous=True,
            is_reparameterized=is_reparameterized,
            use_path_derivative=use_path_derivative,
            group_ndims=group_ndims,
            **kwargs)

    @property
    def mean(self):
        """The mean of the Normal distribution."""
        return self._mean

    @property
    def logstd(self):
        """The log standard deviation of the Normal distribution."""
        return self._logstd

    @property
    def std(self):
        """The standard deviation of the Normal distribution."""
        return self._std

    def _value_shape(self):
        return tf.constant([], dtype=tf.int32)

    def _get_value_shape(self):
        return tf.TensorShape([])

    def _batch_shape(self):
        return tf.broadcast_dynamic_shape(tf.shape(self.mean),
                                          tf.shape(self.std))

    def _get_batch_shape(self):
        return tf.broadcast_static_shape(self.mean.get_shape(),
                                         self.std.get_shape())

    def _sample(self, n_samples):
        mean, std = self.mean, self.std
        if not self.is_reparameterized:
            mean = tf.stop_gradient(mean)
            std = tf.stop_gradient(std)
        shape = tf.concat([[n_samples], self.batch_shape], 0)
        samples = tf.random_normal(shape, dtype=self.dtype) * std + mean
        static_n_samples = n_samples if isinstance(n_samples, int) else None
        samples.set_shape(
            tf.TensorShape([static_n_samples]).concatenate(
                self.get_batch_shape()))
        return samples

    def _log_prob(self, given):
        mean, logstd = self.path_param(self.mean),\
                       self.path_param(self.logstd)
        c = -0.5 * np.log(2 * np.pi)
        precision = tf.exp(-2 * logstd)
        if self._check_numerics:
            precision = tf.check_numerics(precision, "precision")
        return c - logstd - 0.5 * precision * tf.square(given - mean)

    def _prob(self, given):
        return tf.exp(self._log_prob(given))


class FoldNormal(Distribution):
    """
    The class of univariate FoldNormal distribution.
    See :class:`~zhusuan.distributions.base.Distribution` for details.

    .. warning::

        The order of arguments `logstd`/`std` has changed to `std`/`logstd`
        since 0.3.1. Please use named arguments:
        ``FoldNormal(mean, std=..., ...)`` or
        ``FoldNormal(mean, logstd=..., ...)``.

    :param mean: A `float` Tensor. The mean of the FoldNormal distribution.
        Should be broadcastable to match `std` or `logstd`.
    :param _sentinel: Used to prevent positional parameters. Internal,
        do not use.
    :param std: A `float` Tensor. The standard deviation of the FoldNormal
        distribution. Should be positive and broadcastable to match `mean`.
    :param logstd: A `float` Tensor. The log standard deviation of the
        FoldNormal distribution. Should be broadcastable to match `mean`.
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
                 mean=0.,
                 _sentinel=None,
                 std=None,
                 logstd=None,
                 group_ndims=0,
                 is_reparameterized=True,
                 use_path_derivative=False,
                 check_numerics=False,
                 **kwargs):
        ensure_logstd_std_order_change("FoldNormal", _sentinel)
        self._mean = tf.convert_to_tensor(mean)

        if (logstd is None) == (std is None):
            raise ValueError("Either std or logstd should be passed but not "
                             "both of them.")
        elif logstd is None:
            self._std = tf.convert_to_tensor(std)
            dtype = assert_same_float_dtype([(self._mean, 'FoldNormal.mean'),
                                             (self._std, 'FoldNormal.std')])
            logstd = tf.log(self._std)
            if check_numerics:
                logstd = tf.check_numerics(logstd, "log(std)")
            self._logstd = logstd
        else:
            # std is None
            self._logstd = tf.convert_to_tensor(logstd)
            dtype = assert_same_float_dtype(
                [(self._mean, 'FoldNormal.mean'),
                 (self._logstd, 'FoldNormal.logstd')])
            std = tf.exp(self._logstd)
            if check_numerics:
                std = tf.check_numerics(std, "exp(logstd)")
            self._std = std

        try:
            tf.broadcast_static_shape(self._mean.get_shape(),
                                      self._std.get_shape())
        except ValueError:
            raise ValueError(
                "mean and std/logstd should be broadcastable to match each "
                "other. ({} vs. {})".format(
                    self._mean.get_shape(), self._std.get_shape()))
        self._check_numerics = check_numerics
        super(FoldNormal, self).__init__(
            dtype=dtype,
            param_dtype=dtype,
            is_continuous=True,
            is_reparameterized=is_reparameterized,
            use_path_derivative=use_path_derivative,
            group_ndims=group_ndims,
            **kwargs)

    @property
    def mean(self):
        """The mean of the FoldNormal distribution."""
        return self._mean

    @property
    def logstd(self):
        """The log standard deviation of the FoldNormal distribution."""
        return self._logstd

    @property
    def std(self):
        """The standard deviation of the FoldNormal distribution."""
        return self._std

    def _value_shape(self):
        return tf.constant([], dtype=tf.int32)

    def _get_value_shape(self):
        return tf.TensorShape([])

    def _batch_shape(self):
        return tf.broadcast_dynamic_shape(tf.shape(self.mean),
                                          tf.shape(self.std))

    def _get_batch_shape(self):
        return tf.broadcast_static_shape(self.mean.get_shape(),
                                         self.std.get_shape())

    def _sample(self, n_samples):
        mean, std = self.mean, self.std
        if not self.is_reparameterized:
            mean = tf.stop_gradient(mean)
            std = tf.stop_gradient(std)
        shape = tf.concat([[n_samples], self.batch_shape], 0)
        samples = tf.random_normal(shape, dtype=self.dtype) * std + mean
        static_n_samples = n_samples if isinstance(n_samples, int) else None
        samples.set_shape(
            tf.TensorShape([static_n_samples]).concatenate(
                self.get_batch_shape()))
        return samples

    def _log_prob(self, given):
        mean, logstd = self.path_param(self.mean), \
                       self.path_param(self.logstd)
        c = -0.5 * (np.log(2.0) + np.log(np.pi))
        precision = tf.exp(-2.0 * logstd)
        if self._check_numerics:
            precision = tf.check_numerics(precision, "precision")
        mask = tf.log(tf.cast(given >= 0., dtype=precision.dtype))
        return (c - (logstd + 0.5 * precision * tf.square(given - mean)) +
                tf.nn.softplus(-2.0 * mean * given * precision)) + mask

    def _prob(self, given):
        return tf.exp(self._log_prob(given))


class Bernoulli(Distribution):
    """
    The class of univariate Bernoulli distribution.
    See :class:`~zhusuan.distributions.base.Distribution` for details.

    :param logits: A `float` Tensor. The log-odds of probabilities of being 1.

        .. math:: \\mathrm{logits} = \\log \\frac{p}{1 - p}

    :param dtype: The value type of samples from the distribution. Can be
        int (`tf.int16`, `tf.int32`, `tf.int64`) or float (`tf.float16`,
        `tf.float32`, `tf.float64`). Default is `int32`.
    :param group_ndims: A 0-D `int32` Tensor representing the number of
        dimensions in `batch_shape` (counted from the end) that are grouped
        into a single event, so that their probabilities are calculated
        together. Default is 0, which means a single value is an event.
        See :class:`~zhusuan.distributions.base.Distribution` for more detailed
        explanation.
    """

    def __init__(self, logits, dtype=tf.int32, group_ndims=0, **kwargs):
        self._logits = tf.convert_to_tensor(logits)
        param_dtype = assert_same_float_dtype(
            [(self._logits, 'Bernoulli.logits')])

        assert_dtype_is_int_or_float(dtype)

        super(Bernoulli, self).__init__(
            dtype=dtype,
            param_dtype=param_dtype,
            is_continuous=False,
            is_reparameterized=False,
            group_ndims=group_ndims,
            **kwargs)

    @property
    def logits(self):
        """The log-odds of probabilities of being 1."""
        return self._logits

    def _value_shape(self):
        return tf.constant([], dtype=tf.int32)

    def _get_value_shape(self):
        return tf.TensorShape([])

    def _batch_shape(self):
        return tf.shape(self.logits)

    def _get_batch_shape(self):
        return self.logits.get_shape()

    def _sample(self, n_samples):
        p = tf.sigmoid(self.logits)
        shape = tf.concat([[n_samples], self.batch_shape], 0)
        alpha = tf.random_uniform(
            shape, minval=0, maxval=1, dtype=self.param_dtype)
        samples = tf.cast(tf.less(alpha, p), dtype=self.dtype)
        static_n_samples = n_samples if isinstance(n_samples, int) else None
        samples.set_shape(
            tf.TensorShape([static_n_samples]).concatenate(
                self.get_batch_shape()))
        return samples

    def _log_prob(self, given):
        given = tf.cast(given, self.param_dtype)
        given, logits = maybe_explicit_broadcast(
            given, self.logits, 'given', 'logits')
        return -tf.nn.sigmoid_cross_entropy_with_logits(labels=given,
                                                        logits=logits)

    def _prob(self, given):
        return tf.exp(self._log_prob(given))


class Categorical(Distribution):
    """
    The class of univariate Categorical distribution.
    See :class:`~zhusuan.distributions.base.Distribution` for details.

    :param logits: A N-D (N >= 1) `float32` or `float64` Tensor of shape (...,
        n_categories). Each slice `[i, j,..., k, :]` represents the
        un-normalized log probabilities for all categories.

        .. math:: \\mathrm{logits} \\propto \\log p

    :param dtype: The value type of samples from the distribution. Can be
        `float32`, `float64`, `int32`, or `int64`. Default is `int32`.
    :param group_ndims: A 0-D `int32` Tensor representing the number of
        dimensions in `batch_shape` (counted from the end) that are grouped
        into a single event, so that their probabilities are calculated
        together. Default is 0, which means a single value is an event.
        See :class:`~zhusuan.distributions.base.Distribution` for more detailed
        explanation.

    A single sample is a (N-1)-D Tensor with `tf.int32` values in range
    [0, n_categories).
    """

    def __init__(self, logits, dtype=tf.int32, group_ndims=0, **kwargs):
        self._logits = tf.convert_to_tensor(logits)
        param_dtype = assert_same_dtype_in(
            [(self._logits, 'Categorical.logits')],
            [tf.float32, tf.float64])

        allowed_dtypes = [tf.float32, tf.float64, tf.int32, tf.int64]
        assert_dtype_in_dtypes(dtype, allowed_dtypes)

        self._logits = assert_rank_at_least_one(
                self._logits, 'Categorical.logits')
        self._n_categories = get_shape_at(self._logits, -1)

        super(Categorical, self).__init__(
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
        return tf.constant([], dtype=tf.int32)

    def _get_value_shape(self):
        return tf.TensorShape([])

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
        samples_flat = tf.cast(samples_flat, self.dtype)
        if self.logits.get_shape().ndims == 2:
            return samples_flat
        shape = tf.concat([[n_samples], self.batch_shape], 0)
        samples = tf.reshape(samples_flat, shape)
        static_n_samples = n_samples if isinstance(n_samples, int) else None
        samples.set_shape(
            tf.TensorShape([static_n_samples]).concatenate(
                self.get_batch_shape()))
        return samples

    def _log_prob(self, given):
        logits = self.logits

        def _broadcast(given, logits):
            # static shape has been checked in base class.
            ones_ = tf.ones(tf.shape(logits)[:-1], self.dtype)
            if logits.get_shape():
                ones_.set_shape(logits.get_shape()[:-1])
            given *= ones_
            logits *= tf.ones_like(tf.expand_dims(given, -1), self.param_dtype)
            return given, logits

        # def _is_same_dynamic_shape(given, logits):
        #     return tf.cond(
        #         tf.equal(tf.rank(given), tf.rank(logits) - 1),
        #         lambda: tf.reduce_all(tf.equal(
        #             tf.concat([tf.shape(given), tf.shape(logits)[:-1]], 0),
        #             tf.concat([tf.shape(logits)[:-1], tf.shape(given)], 0))),
        #         lambda: tf.convert_to_tensor(False, tf.bool))

        if not (given.get_shape() and logits.get_shape()):
            given, logits = _broadcast(given, logits)
        else:
            if given.get_shape().ndims != logits.get_shape().ndims - 1:
                given, logits = _broadcast(given, logits)
            elif given.get_shape().is_fully_defined() and \
                    logits.get_shape()[:-1].is_fully_defined():
                if given.get_shape() != logits.get_shape()[:-1]:
                    given, logits = _broadcast(given, logits)
            else:
                # Below code seems to induce a BUG when this function is
                # called in HMC. Probably due to tensorflow's not supporting
                # control flow edge from an op inside the body to outside.
                # We should further fix this.
                #
                # given, logits = tf.cond(
                #     is_same_dynamic_shape(given, logits),
                #     lambda: (given, logits),
                #     lambda: _broadcast(given, logits, 'given', 'logits'))
                given, logits = _broadcast(given, logits)

        # `labels` type of `sparse_softmax_cross_entropy_with_logits` must be
        # int32 or int64
        if self.dtype == tf.float32:
            given = tf.to_int32(given)
        elif self.dtype == tf.float64:
            given = tf.to_int64(given)
        log_p = -tf.nn.sparse_softmax_cross_entropy_with_logits(labels=given,
                                                                logits=logits)
        if given.get_shape() and logits.get_shape():
            log_p.set_shape(tf.broadcast_static_shape(given.get_shape(),
                                                      logits.get_shape()[:-1]))
        return log_p

    def _prob(self, given):
        return tf.exp(self._log_prob(given))


Discrete = Categorical


class Uniform(Distribution):
    """
    The class of univariate Uniform distribution.
    See :class:`~zhusuan.distributions.base.Distribution` for details.

    :param minval: A `float` Tensor. The lower bound on the range of the
        uniform distribution. Should be broadcastable to match `maxval`.
    :param maxval: A `float` Tensor. The upper bound on the range of the
        uniform distribution. Should be element-wise bigger than `minval`.
    :param group_ndims: A 0-D `int32` Tensor representing the number of
        dimensions in `batch_shape` (counted from the end) that are grouped
        into a single event, so that their probabilities are calculated
        together. Default is 0, which means a single value is an event.
        See :class:`~zhusuan.distributions.base.Distribution` for more detailed
        explanation.
    :param is_reparameterized: A Bool. If True, gradients on samples from this
        distribution are allowed to propagate into inputs, using the
        reparametrization trick from (Kingma, 2013).
    :param check_numerics: Bool. Whether to check numeric issues.
    """

    def __init__(self,
                 minval=0.,
                 maxval=1.,
                 group_ndims=0,
                 is_reparameterized=True,
                 check_numerics=False,
                 **kwargs):
        self._minval = tf.convert_to_tensor(minval)
        self._maxval = tf.convert_to_tensor(maxval)
        dtype = assert_same_float_dtype(
            [(self._minval, 'Uniform.minval'),
             (self._maxval, 'Uniform.maxval')])

        try:
            tf.broadcast_static_shape(self._minval.get_shape(),
                                      self._maxval.get_shape())
        except ValueError:
            raise ValueError(
                "minval and maxval should be broadcastable to match each "
                "other. ({} vs. {})".format(
                    self._minval.get_shape(), self._maxval.get_shape()))
        self._check_numerics = check_numerics
        super(Uniform, self).__init__(
            dtype=dtype,
            param_dtype=dtype,
            is_continuous=True,
            is_reparameterized=is_reparameterized,
            group_ndims=group_ndims,
            **kwargs)

    @property
    def minval(self):
        """The lower bound on the range of the uniform distribution."""
        return self._minval

    @property
    def maxval(self):
        """The upper bound on the range of the uniform distribution."""
        return self._maxval

    def _value_shape(self):
        return tf.constant([], tf.int32)

    def _get_value_shape(self):
        return tf.TensorShape([])

    def _batch_shape(self):
        return tf.broadcast_dynamic_shape(tf.shape(self.minval),
                                          tf.shape(self.maxval))

    def _get_batch_shape(self):
        return tf.broadcast_static_shape(self.minval.get_shape(),
                                         self.maxval.get_shape())

    def _sample(self, n_samples):
        minval, maxval = self.minval, self.maxval
        if not self.is_reparameterized:
            minval = tf.stop_gradient(minval)
            maxval = tf.stop_gradient(maxval)
        shape = tf.concat([[n_samples], self.batch_shape], 0)
        samples = tf.random_uniform(shape, 0, 1, dtype=self.dtype) * \
            (maxval - minval) + minval
        static_n_samples = n_samples if isinstance(n_samples, int) else None
        samples.set_shape(
            tf.TensorShape([static_n_samples]).concatenate(
                self.get_batch_shape()))
        return samples

    def _log_prob(self, given):
        log_p = tf.log(self._prob(given))
        if self._check_numerics:
            log_p = tf.check_numerics(log_p, "log_p")
        return log_p

    def _prob(self, given):
        mask = tf.cast(tf.logical_and(tf.less_equal(self.minval, given),
                                      tf.less(given, self.maxval)),
                       self.dtype)
        p = 1. / (self.maxval - self.minval)
        if self._check_numerics:
            p = tf.check_numerics(p, "p")
        return p * mask


class Gamma(Distribution):
    """
    The class of univariate Gamma distribution.
    See :class:`~zhusuan.distributions.base.Distribution` for details.

    :param alpha: A `float` Tensor. The shape parameter of the Gamma
        distribution. Should be positive and broadcastable to match `beta`.
    :param beta: A `float` Tensor. The inverse scale parameter of the Gamma
        distribution. Should be positive and broadcastable to match `alpha`.
    :param group_ndims: A 0-D `int32` Tensor representing the number of
        dimensions in `batch_shape` (counted from the end) that are grouped
        into a single event, so that their probabilities are calculated
        together. Default is 0, which means a single value is an event.
        See :class:`~zhusuan.distributions.base.Distribution` for more detailed
        explanation.
    :param check_numerics: Bool. Whether to check numeric issues.
    """

    def __init__(self,
                 alpha,
                 beta,
                 group_ndims=0,
                 check_numerics=False,
                 **kwargs):
        self._alpha = tf.convert_to_tensor(alpha)
        self._beta = tf.convert_to_tensor(beta)
        dtype = assert_same_float_dtype(
            [(self._alpha, 'Gamma.alpha'),
             (self._beta, 'Gamma.beta')])

        try:
            tf.broadcast_static_shape(self._alpha.get_shape(),
                                      self._beta.get_shape())
        except ValueError:
            raise ValueError(
                "alpha and beta should be broadcastable to match each "
                "other. ({} vs. {})".format(
                    self._alpha.get_shape(), self._beta.get_shape()))
        self._check_numerics = check_numerics
        super(Gamma, self).__init__(
            dtype=dtype,
            param_dtype=dtype,
            is_continuous=True,
            is_reparameterized=False,
            group_ndims=group_ndims,
            **kwargs)

    @property
    def alpha(self):
        """The shape parameter of the Gamma distribution."""
        return self._alpha

    @property
    def beta(self):
        """The inverse scale parameter of the Gamma distribution."""
        return self._beta

    def _value_shape(self):
        return tf.constant([], dtype=tf.int32)

    def _get_value_shape(self):
        return tf.TensorShape([])

    def _batch_shape(self):
        return tf.broadcast_dynamic_shape(tf.shape(self.alpha),
                                          tf.shape(self.beta))

    def _get_batch_shape(self):
        return tf.broadcast_static_shape(self.alpha.get_shape(),
                                         self.beta.get_shape())

    def _sample(self, n_samples):
        return tf.random_gamma([n_samples], self.alpha,
                               beta=self.beta, dtype=self.dtype)

    def _log_prob(self, given):
        alpha, beta = self.alpha, self.beta
        log_given = tf.log(given)
        log_beta = tf.log(beta)
        lgamma_alpha = tf.lgamma(alpha)
        if self._check_numerics:
            log_given = tf.check_numerics(log_given, "log(given)")
            log_beta = tf.check_numerics(log_beta, "log(beta)")
            lgamma_alpha = tf.check_numerics(lgamma_alpha, "lgamma(alpha)")
        return alpha * log_beta - lgamma_alpha + (alpha - 1) * log_given - \
            beta * given

    def _prob(self, given):
        return tf.exp(self._log_prob(given))


class Beta(Distribution):
    """
    The class of univariate Beta distribution.
    See :class:`~zhusuan.distributions.base.Distribution` for details.

    :param alpha: A `float` Tensor. One of the two shape parameters of the
        Beta distribution. Should be positive and broadcastable to match
        `beta`.
    :param beta: A `float` Tensor. One of the two shape parameters of the
        Beta distribution. Should be positive and broadcastable to match
        `alpha`.
    :param group_ndims: A 0-D `int32` Tensor representing the number of
        dimensions in `batch_shape` (counted from the end) that are grouped
        into a single event, so that their probabilities are calculated
        together. Default is 0, which means a single value is an event.
        See :class:`~zhusuan.distributions.base.Distribution` for more detailed
        explanation.
    :param check_numerics: Bool. Whether to check numeric issues.
    """

    def __init__(self,
                 alpha,
                 beta,
                 group_ndims=0,
                 check_numerics=False,
                 **kwargs):
        self._alpha = tf.convert_to_tensor(alpha)
        self._beta = tf.convert_to_tensor(beta)
        dtype = assert_same_float_dtype(
            [(self._alpha, 'Beta.alpha'),
             (self._beta, 'Beta.beta')])

        try:
            tf.broadcast_static_shape(self._alpha.get_shape(),
                                      self._beta.get_shape())
        except ValueError:
            raise ValueError(
                "alpha and beta should be broadcastable to match each "
                "other. ({} vs. {})".format(
                    self._alpha.get_shape(), self._beta.get_shape()))
        self._check_numerics = check_numerics
        super(Beta, self).__init__(
            dtype=dtype,
            param_dtype=dtype,
            is_continuous=True,
            is_reparameterized=False,
            group_ndims=group_ndims,
            **kwargs)

    @property
    def alpha(self):
        """One of the two shape parameters of the Beta distribution."""
        return self._alpha

    @property
    def beta(self):
        """One of the two shape parameters of the Beta distribution."""
        return self._beta

    def _value_shape(self):
        return tf.constant([], dtype=tf.int32)

    def _get_value_shape(self):
        return tf.TensorShape([])

    def _batch_shape(self):
        return tf.broadcast_dynamic_shape(tf.shape(self.alpha),
                                          tf.shape(self.beta))

    def _get_batch_shape(self):
        return tf.broadcast_static_shape(self.alpha.get_shape(),
                                         self.beta.get_shape())

    def _sample(self, n_samples):
        alpha, beta = maybe_explicit_broadcast(
            self.alpha, self.beta, 'alpha', 'beta')
        x = tf.random_gamma([n_samples], alpha, beta=1, dtype=self.dtype)
        y = tf.random_gamma([n_samples], beta, beta=1, dtype=self.dtype)
        return x / (x + y)

    def _log_prob(self, given):
        # TODO: not right when given=0 or 1
        alpha, beta = self.alpha, self.beta
        log_given = tf.log(given)
        log_1_minus_given = tf.log(1 - given)
        lgamma_alpha, lgamma_beta = tf.lgamma(alpha), tf.lgamma(beta)
        lgamma_alpha_plus_beta = tf.lgamma(alpha + beta)

        if self._check_numerics:
            log_given = tf.check_numerics(log_given, "log(given)")
            log_1_minus_given = tf.check_numerics(
                log_1_minus_given, "log(1 - given)")
            lgamma_alpha = tf.check_numerics(lgamma_alpha, "lgamma(alpha)")
            lgamma_beta = tf.check_numerics(lgamma_beta, "lgamma(beta)")
            lgamma_alpha_plus_beta = tf.check_numerics(
                lgamma_alpha_plus_beta, "lgamma(alpha + beta)")

        return (alpha - 1) * log_given + (beta - 1) * log_1_minus_given - (
            lgamma_alpha + lgamma_beta - lgamma_alpha_plus_beta)

    def _prob(self, given):
        return tf.exp(self._log_prob(given))


class Poisson(Distribution):
    """
    The class of univariate Poisson distribution.
    See :class:`~zhusuan.distributions.base.Distribution` for details.

    :param rate: A `float` Tensor. The rate parameter of Poisson
        distribution. Must be positive.
    :param dtype: The value type of samples from the distribution. Can be
        int (`tf.int16`, `tf.int32`, `tf.int64`) or float (`tf.float16`,
        `tf.float32`, `tf.float64`). Default is `int32`.
    :param group_ndims: A 0-D `int32` Tensor representing the number of
        dimensions in `batch_shape` (counted from the end) that are grouped
        into a single event, so that their probabilities are calculated
        together. Default is 0, which means a single value is an event.
        See :class:`~zhusuan.distributions.base.Distribution` for more detailed
        explanation.
    :param check_numerics: Bool. Whether to check numeric issues.
    """

    def __init__(self,
                 rate,
                 dtype=tf.int32,
                 group_ndims=0,
                 check_numerics=False,
                 **kwargs):
        self._rate = tf.convert_to_tensor(rate)
        param_dtype = assert_same_float_dtype(
            [(self._rate, 'Poisson.rate')])

        assert_dtype_is_int_or_float(dtype)

        self._check_numerics = check_numerics

        super(Poisson, self).__init__(
            dtype=dtype,
            param_dtype=param_dtype,
            is_continuous=False,
            is_reparameterized=False,
            group_ndims=group_ndims,
            **kwargs)

    @property
    def rate(self):
        """The rate parameter of Poisson."""
        return self._rate

    def _value_shape(self):
        return tf.constant([], dtype=tf.int32)

    def _get_value_shape(self):
        return tf.TensorShape([])

    def _batch_shape(self):
        return tf.shape(self.rate)

    def _get_batch_shape(self):
        return self.rate.get_shape()

    def _sample(self, n_samples):
        samples = tf.random_poisson(self.rate, [n_samples],
                                    dtype=self.param_dtype)
        if self.param_dtype != self.dtype:
            samples = tf.cast(samples, self.dtype)
        return samples

    def _log_prob(self, given):
        rate = self.rate
        given = tf.cast(given, self.param_dtype)

        log_rate = tf.log(rate)
        lgamma_given_plus_1 = tf.lgamma(given + 1)

        if self._check_numerics:
            log_rate = tf.check_numerics(log_rate, "log(rate)")
            lgamma_given_plus_1 = tf.check_numerics(
                lgamma_given_plus_1, "lgamma(given + 1)")
        return given * log_rate - rate - lgamma_given_plus_1

    def _prob(self, given):
        return tf.exp(self._log_prob(given))


class Binomial(Distribution):
    """
    The class of univariate Binomial distribution.
    See :class:`~zhusuan.distributions.base.Distribution` for details.

    :param logits: A `float` Tensor. The log-odds of probabilities.

        .. math:: \\mathrm{logits} = \\log \\frac{p}{1 - p}

    :param n_experiments: A 0-D `int32` Tensor. The number of experiments
        for each sample.
    :param dtype: The value type of samples from the distribution. Can be
        int (`tf.int16`, `tf.int32`, `tf.int64`) or float (`tf.float16`,
        `tf.float32`, `tf.float64`). Default is `int32`.
    :param group_ndims: A 0-D `int32` Tensor representing the number of
        dimensions in `batch_shape` (counted from the end) that are grouped
        into a single event, so that their probabilities are calculated
        together. Default is 0, which means a single value is an event.
        See :class:`~zhusuan.distributions.base.Distribution` for more detailed
        explanation.
    :param check_numerics: Bool. Whether to check numeric issues.
    """

    def __init__(self,
                 logits,
                 n_experiments,
                 dtype=tf.int32,
                 group_ndims=0,
                 check_numerics=False,
                 **kwargs):
        self._logits = tf.convert_to_tensor(logits)
        param_dtype = assert_same_float_dtype(
            [(self._logits, 'Binomial.logits')])

        assert_dtype_is_int_or_float(dtype)

        sign_err_msg = "n_experiments must be positive"
        if isinstance(n_experiments, int):
            if n_experiments <= 0:
                raise ValueError(sign_err_msg)
            self._n_experiments = n_experiments
        else:
            try:
                n_experiments = tf.convert_to_tensor(n_experiments, tf.int32)
            except ValueError:
                raise TypeError('n_experiments must be int32')
            _assert_rank_op = tf.assert_rank(
                n_experiments, 0,
                message="n_experiments should be a scalar (0-D Tensor).")
            _assert_positive_op = tf.assert_greater(
                n_experiments, 0, message=sign_err_msg)
            with tf.control_dependencies([_assert_rank_op,
                                          _assert_positive_op]):
                self._n_experiments = tf.identity(n_experiments)

        self._check_numerics = check_numerics
        super(Binomial, self).__init__(
            dtype=dtype,
            param_dtype=param_dtype,
            is_continuous=False,
            is_reparameterized=False,
            group_ndims=group_ndims,
            **kwargs)

    @property
    def n_experiments(self):
        """The number of experiments."""
        return self._n_experiments

    @property
    def logits(self):
        """The log-odds of probabilities."""
        return self._logits

    def _value_shape(self):
        return tf.constant([], dtype=tf.int32)

    def _get_value_shape(self):
        return tf.TensorShape([])

    def _batch_shape(self):
        return tf.shape(self.logits)

    def _get_batch_shape(self):
        return self.logits.get_shape()

    def _sample(self, n_samples):
        n = self.n_experiments
        if self.logits.get_shape().ndims == 1:
            logits_flat = self.logits
        else:
            logits_flat = tf.reshape(self.logits, [-1])
        log_1_minus_p = -tf.nn.softplus(logits_flat)
        log_p = logits_flat + log_1_minus_p
        stacked_logits_flat = tf.stack([log_1_minus_p, log_p], axis=-1)
        samples_flat = tf.transpose(
            tf.multinomial(stacked_logits_flat, n_samples * n))

        shape = tf.concat([[n, n_samples], self.batch_shape], 0)
        samples = tf.reduce_sum(tf.reshape(samples_flat, shape), axis=0)

        static_n_samples = n_samples if isinstance(n_samples, int) else None
        static_shape = tf.TensorShape([static_n_samples]).concatenate(
            self.get_batch_shape())
        samples.set_shape(static_shape)

        return tf.cast(samples, self.dtype)

    def _log_prob(self, given):
        logits = self.logits
        n = tf.cast(self.n_experiments, self.param_dtype)
        given = tf.cast(given, self.param_dtype)

        log_1_minus_p = -tf.nn.softplus(logits)
        lgamma_n_plus_1 = tf.lgamma(n + 1)
        lgamma_given_plus_1 = tf.lgamma(given + 1)
        lgamma_n_minus_given_plus_1 = tf.lgamma(n - given + 1)

        if self._check_numerics:
            lgamma_given_plus_1 = tf.check_numerics(
                lgamma_given_plus_1, "lgamma(given + 1)")
            lgamma_n_minus_given_plus_1 = tf.check_numerics(
                lgamma_n_minus_given_plus_1, "lgamma(n - given + 1)")

        return lgamma_n_plus_1 - lgamma_n_minus_given_plus_1 - \
            lgamma_given_plus_1 + given * logits + n * log_1_minus_p

    def _prob(self, given):
        return tf.exp(self._log_prob(given))


class InverseGamma(Distribution):
    """
    The class of univariate InverseGamma distribution.
    See :class:`~zhusuan.distributions.base.Distribution` for details.

    :param alpha: A `float` Tensor. The shape parameter of the InverseGamma
        distribution. Should be positive and broadcastable to match `beta`.
    :param beta: A `float` Tensor. The scale parameter of the InverseGamma
        distribution. Should be positive and broadcastable to match `alpha`.
    :param group_ndims: A 0-D `int32` Tensor representing the number of
        dimensions in `batch_shape` (counted from the end) that are grouped
        into a single event, so that their probabilities are calculated
        together. Default is 0, which means a single value is an event.
        See :class:`~zhusuan.distributions.base.Distribution` for more detailed
        explanation.
    :param check_numerics: Bool. Whether to check numeric issues.
    """

    def __init__(self,
                 alpha,
                 beta,
                 group_ndims=0,
                 check_numerics=False,
                 **kwargs):
        self._alpha = tf.convert_to_tensor(alpha)
        self._beta = tf.convert_to_tensor(beta)
        dtype = assert_same_float_dtype(
            [(self._alpha, 'InverseGamma.alpha'),
             (self._beta, 'InverseGamma.beta')])

        try:
            tf.broadcast_static_shape(self._alpha.get_shape(),
                                      self._beta.get_shape())
        except ValueError:
            raise ValueError(
                "alpha and beta should be broadcastable to match each "
                "other. ({} vs. {})".format(
                    self._alpha.get_shape(), self._beta.get_shape()))
        self._check_numerics = check_numerics
        super(InverseGamma, self).__init__(
            dtype=dtype,
            param_dtype=dtype,
            is_continuous=True,
            is_reparameterized=False,
            group_ndims=group_ndims,
            **kwargs)

    @property
    def alpha(self):
        """The shape parameter of the InverseGamma distribution."""
        return self._alpha

    @property
    def beta(self):
        """The scale parameter of the InverseGamma distribution."""
        return self._beta

    def _value_shape(self):
        return tf.constant([], dtype=tf.int32)

    def _get_value_shape(self):
        return tf.TensorShape([])

    def _batch_shape(self):
        return tf.broadcast_dynamic_shape(tf.shape(self.alpha),
                                          tf.shape(self.beta))

    def _get_batch_shape(self):
        return tf.broadcast_static_shape(self.alpha.get_shape(),
                                         self.beta.get_shape())

    def _sample(self, n_samples):
        gamma = tf.random_gamma([n_samples], self.alpha,
                                beta=self.beta, dtype=self.dtype)
        return 1 / gamma

    def _log_prob(self, given):
        alpha, beta = self.alpha, self.beta
        log_given = tf.log(given)
        log_beta = tf.log(beta)
        lgamma_alpha = tf.lgamma(alpha)

        if self._check_numerics:
            log_given = tf.check_numerics(log_given, "log(given)")
            log_beta = tf.check_numerics(log_beta, "log(beta)")
            lgamma_alpha = tf.check_numerics(lgamma_alpha, "lgamma(alpha)")

        return alpha * log_beta - lgamma_alpha - (alpha + 1) * log_given - \
            beta / given

    def _prob(self, given):
        return tf.exp(self._log_prob(given))


class Laplace(Distribution):
    """
    The class of univariate Laplace distribution.
    See :class:`~zhusuan.distributions.base.Distribution` for details.

    :param loc: A `float` Tensor. The location parameter of the Laplace
        distribution. Should be broadcastable to match `scale`.
    :param scale: A `float` Tensor. The scale parameter of the Laplace
        distribution. Should be positive and broadcastable to match `loc`.
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
                 loc,
                 scale,
                 group_ndims=0,
                 is_reparameterized=True,
                 use_path_derivative=False,
                 check_numerics=False,
                 **kwargs):
        self._loc = tf.convert_to_tensor(loc)
        self._scale = tf.convert_to_tensor(scale)
        dtype = assert_same_float_dtype(
            [(self._loc, 'Laplace.loc'),
             (self._scale, 'Laplace.scale')])

        try:
            tf.broadcast_static_shape(self._loc.get_shape(),
                                      self._scale.get_shape())
        except ValueError:
            raise ValueError(
                "loc and scale should be broadcastable to match each "
                "other. ({} vs. {})".format(
                    self._loc.get_shape(), self._scale.get_shape()))
        self._check_numerics = check_numerics
        super(Laplace, self).__init__(
            dtype=dtype,
            param_dtype=dtype,
            is_continuous=True,
            is_reparameterized=is_reparameterized,
            use_path_derivative=use_path_derivative,
            group_ndims=group_ndims,
            **kwargs)

    @property
    def loc(self):
        """The location parameter of the Laplace distribution."""
        return self._loc

    @property
    def scale(self):
        """The scale parameter of the Laplace distribution."""
        return self._scale

    def _value_shape(self):
        return tf.constant([], dtype=tf.int32)

    def _get_value_shape(self):
        return tf.TensorShape([])

    def _batch_shape(self):
        return tf.broadcast_dynamic_shape(tf.shape(self.loc),
                                          tf.shape(self.scale))

    def _get_batch_shape(self):
        return tf.broadcast_static_shape(self.loc.get_shape(),
                                         self.scale.get_shape())

    def _sample(self, n_samples):
        # samples must be sampled from (-1, 1) rather than [-1, 1)
        loc, scale = self.loc, self.scale
        if not self.is_reparameterized:
            loc = tf.stop_gradient(loc)
            scale = tf.stop_gradient(scale)
        shape = tf.concat([[n_samples], self.batch_shape], 0)
        uniform_samples = tf.random_uniform(
            shape=shape,
            minval=np.nextafter(self.dtype.as_numpy_dtype(-1.),
                                self.dtype.as_numpy_dtype(0.)),
            maxval=1.,
            dtype=self.dtype)
        samples = loc - scale * tf.sign(uniform_samples) * \
            tf.log1p(-tf.abs(uniform_samples))
        static_n_samples = n_samples if isinstance(n_samples, int) else None
        samples.set_shape(
            tf.TensorShape([static_n_samples]).concatenate(
                self.get_batch_shape()))
        return samples

    def _log_prob(self, given):
        loc, scale = self.path_param(self.loc),\
                     self.path_param(self.scale)
        log_scale = tf.log(scale)
        if self._check_numerics:
            log_scale = tf.check_numerics(log_scale, "log(scale)")
        return -np.log(2.) - log_scale - tf.abs(given - loc) / scale

    def _prob(self, given):
        return tf.exp(self._log_prob(given))


class BinConcrete(Distribution):
    """
    The class of univariate BinConcrete distribution from (Maddison, 2016).
    It is the binary case of
    :class:`~zhusuan.distributions.multivariate.Concrete`.
    See :class:`~zhusuan.distributions.base.Distribution` for details.

    .. seealso::

        :class:`~zhusuan.distributions.multivariate.Concrete` and
        :class:`~zhusuan.distributions.multivariate.ExpConcrete`

    :param temperature: A 0-D `float` Tensor. The temperature of the relaxed
        distribution. The temperature should be positive.
    :param logits: A `float` Tensor. The log-odds of probabilities of being 1.

        .. math:: \\mathrm{logits} = \\log \\frac{p}{1 - p}

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
            [(self._logits, 'BinConcrete.logits'),
             (self._temperature, 'BinConcrete.temperature')])

        self._temperature = assert_scalar(
            self._temperature, 'BinConcrete.temperature')

        self._check_numerics = check_numerics
        super(BinConcrete, self).__init__(
            dtype=param_dtype,
            param_dtype=param_dtype,
            is_continuous=True,
            is_reparameterized=is_reparameterized,
            use_path_derivative=use_path_derivative,
            group_ndims=group_ndims,
            **kwargs)

    @property
    def temperature(self):
        """The temperature of BinConcrete."""
        return self._temperature

    @property
    def logits(self):
        """The log-odds of probabilities."""
        return self._logits

    def _value_shape(self):
        return tf.constant([], dtype=tf.int32)

    def _get_value_shape(self):
        return tf.TensorShape([])

    def _batch_shape(self):
        return tf.shape(self.logits)

    def _get_batch_shape(self):
        return self.logits.get_shape()

    def _sample(self, n_samples):
        logits, temperature = self.logits, self.temperature
        if not self.is_reparameterized:
            logits = tf.stop_gradient(logits)
            temperature = tf.stop_gradient(temperature)
        shape = tf.concat([[n_samples], self.batch_shape], 0)

        uniform = open_interval_standard_uniform(shape, self.dtype)
        # TODO: add Logistic distribution
        logistic = tf.log(uniform) - tf.log(1 - uniform)
        samples = tf.sigmoid((logits + logistic) / temperature)

        static_n_samples = n_samples if isinstance(n_samples, int) else None
        samples.set_shape(
            tf.TensorShape([static_n_samples]).concatenate(
                self.get_batch_shape()))
        return samples

    def _log_prob(self, given):
        temperature, logits = self.path_param(self.temperature), \
                              self.path_param(self.logits)
        log_given = tf.log(given)
        log_1_minus_given = tf.log(1 - given)
        log_temperature = tf.log(temperature)

        if self._check_numerics:
            log_given = tf.check_numerics(log_given, "log(given)")
            log_1_minus_given = tf.check_numerics(
                log_1_minus_given, "log(1 - given)")
            log_temperature = tf.check_numerics(
                log_temperature, "log(temperature)")

        logistic_given = log_given - log_1_minus_given
        temp = temperature * logistic_given - logits

        return log_temperature - log_given - log_1_minus_given + \
            temp - 2 * tf.nn.softplus(temp)

    def _prob(self, given):
        return tf.exp(self._log_prob(given))


BinGumbelSoftmax = BinConcrete
