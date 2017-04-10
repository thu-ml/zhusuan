#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np

from zhusuan.utils import convert_to_int, add_name_scope


__all__ = [
    'uniform',
    'norm',
    'logistic',
    'bernoulli',
    'discrete',
]


class Uniform(object):
    """
    Class of Uniform distribution.
    """

    def __init__(self):
        pass

    @add_name_scope
    def rvs(self,
            minval=0.,
            maxval=1.,
            sample_dim=None,
            n_samples=1,
            reparameterized=False):
        """
        Generate independent uniform samples which forms a Tensor.
        The lower bound `minval` is included in the range, while the upper
        bound `maxval` is excluded.

        :param minval: A Tensor, python value, or numpy array. The lower bound
            on the range of the uniform distribution.
        :param maxval: A Tensor, python value, or numpy array. The upper bound
            on the range of the uniform distribution. Should have the same
            shape with and element-wise bigger than `minval`.
        :param sample_dim: A Tensor scalar, int, or None. The sample dimension.
            If None, this means no new sample dimension is created. In this
            case `n_samples` must be set to 1, otherwise an Exception is
            raised.
        :param n_samples: A Tensor scalar or int. Number of samples to
            generate.
        :param reparameterized: Bool. If True, gradients on samples from this
            Uniform distribution are allowed to propagate into `minval` and
            `maxval`.
        :param check_numerics: Bool. Whether to check numeric issues.

        :return: A Tensor. Samples from the Uniform distribution.
        """
        static_sample_dim = convert_to_int(sample_dim)
        if static_sample_dim and not (static_sample_dim >= 0):
            raise ValueError("Only support non-negative sample_dim ({})".
                             format(static_sample_dim))
        static_n_samples = convert_to_int(n_samples)
        minval = tf.convert_to_tensor(minval, dtype=tf.float32)
        maxval = tf.convert_to_tensor(maxval, dtype=tf.float32)
        n_samples = tf.convert_to_tensor(n_samples, dtype=tf.int32)
        try:
            static_base_shape = minval.get_shape().merge_with(
                maxval.get_shape())
        except ValueError:
            raise ValueError(
                "minval and maxval must have the same shape (%s vs %s)"
                % (minval.get_shape(), maxval.get_shape()))
        static_base_shape = static_base_shape.as_list()
        _assert_shape_match = tf.assert_equal(
            tf.shape(minval), tf.shape(maxval),
            message="minval and maxval must have the same shape")
        with tf.control_dependencies([_assert_shape_match]):
            base_shape = tf.shape(minval)
        if not reparameterized:
            minval = tf.stop_gradient(minval)
            maxval = tf.stop_gradient(maxval)
        if sample_dim is None:
            _assert_one_sample = tf.assert_equal(
                n_samples, 1,
                message="n_samples must be 1 when sample_dim is None")
            with tf.control_dependencies([_assert_one_sample]):
                shape = tf.identity(base_shape)
                static_shape = static_base_shape
        else:
            sample_dim = tf.convert_to_tensor(sample_dim, tf.int32)
            _assert_positive_dim = tf.assert_greater_equal(
                sample_dim, 0, message="Only support non-negative sample_dim")
            with tf.control_dependencies([_assert_positive_dim]):
                sample_dim = tf.identity(sample_dim)
            shape = tf.concat([base_shape[:sample_dim],
                               tf.stack([n_samples]),
                               base_shape[sample_dim:]], 0)
            if static_sample_dim is not None:
                static_shape = static_base_shape[:static_sample_dim] + \
                    [static_n_samples] + static_base_shape[static_sample_dim:]
            else:
                static_shape = [None] * (len(static_base_shape) + 1)
            minval = tf.expand_dims(minval, sample_dim)
            maxval = tf.expand_dims(maxval, sample_dim)
        samples = tf.random_uniform(shape, 0, 1) * (maxval - minval) + minval
        samples.set_shape(static_shape)
        return samples

    @add_name_scope
    def logpdf(self, x, minval=0., maxval=1., sample_dim=None):
        """
        Log probability density function of Uniform distribution.
        The lower bound `minval` is included in the range, while the upper
        bound `maxval` is excluded.

        :param x: A Tensor, python value, or numpy array. The value at which to
            evaluate the log density function. Can be broadcast to match
            `minval` and `maxval`.
        :param minval: A Tensor, python value, or numpy array. The lower bound
            on the range of the uniform distribution. Can be broadcast to
            match `x` and `maxval`.
        :param maxval: A Tensor, python value, or numpy array. The upper bound
            on the range of the uniform distribution. Can be broadcast to
            match `x` and `minval` and should be element-wise bigger than
            `minval`.
        :param sample_dim: A Tensor scalar, int, or None. The sample dimension
            which `x` has more than `minval` and `maxval`. If None, `x` is
            supposed to be a one-sample result, which remains the same shape
            with `minval` and `maxval` (after broadcast).

        :return: A Tensor of the same shape as `x` (after broadcast) with
            function values.
        """
        # TODO: check maxval > minval
        static_sample_dim = convert_to_int(sample_dim)
        if static_sample_dim and not (static_sample_dim >= 0):
            raise ValueError("Only support non-negative sample_dim ({})".
                             format(static_sample_dim))
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        minval = tf.convert_to_tensor(minval, dtype=tf.float32)
        maxval = tf.convert_to_tensor(maxval, dtype=tf.float32)
        if sample_dim is not None:
            sample_dim = tf.convert_to_tensor(sample_dim, tf.int32)
            _assert_positive_dim = tf.assert_greater_equal(
                sample_dim, 0, message="Only support non-negative sample_dim")
            with tf.control_dependencies([_assert_positive_dim]):
                sample_dim = tf.identity(sample_dim)
            minval_shape = minval.get_shape().as_list()
            maxval_shape = maxval.get_shape().as_list()
            minval = tf.expand_dims(minval, sample_dim)
            maxval = tf.expand_dims(maxval, sample_dim)
            if static_sample_dim is not None:
                minval.set_shape(minval_shape[:static_sample_dim] + [1] +
                                 minval_shape[static_sample_dim:])
                maxval.set_shape(maxval_shape[:static_sample_dim] + [1] +
                                 maxval_shape[static_sample_dim:])
        mask = tf.cast(tf.logical_and(tf.less_equal(minval, x),
                                      tf.less(x, maxval)),
                       tf.float32)
        return tf.log(1. / (maxval - minval) * mask)


class Normal(object):
    """
    Class of Normal distribution.
    """

    def __init__(self):
        pass

    @add_name_scope
    def rvs(self,
            mean=0.,
            logstd=0.,
            sample_dim=None,
            n_samples=1,
            reparameterized=False,
            check_numerics=True):
        """
        Generate independent Normal samples which forms a Tensor.

        :param mean: A Tensor, python value, or numpy array. The mean of the
            Normal distribution.
        :param logstd: A Tensor, python value, or numpy array. The log
            standard deviation of the Normal distribution. Should have the same
            shape with `mean`.
        :param sample_dim: A Tensor scalar, int, or None. The sample dimension.
            If None, this means no new sample dimension is created. In this
            case `n_samples` must be set to 1, otherwise an Exception is
            raised.
        :param n_samples: A Tensor scalar or int. Number of samples to
            generate.
        :param reparameterized: Bool. If True, gradients on samples from this
            Normal distribution are allowed to propagate into inputs in this
            function, using the reparametrization trick from (Kingma, 2013).
        :param check_numerics: Bool. Whether to check numeric issues.

        :return: A Tensor. Samples from the Normal distribution.
        """
        # TODO: as_list() fails when static_shape unknown
        static_sample_dim = convert_to_int(sample_dim)
        if static_sample_dim and not (static_sample_dim >= 0):
            raise ValueError("Only support non-negative sample_dim ({})".
                             format(static_sample_dim))
        static_n_samples = convert_to_int(n_samples)
        mean = tf.convert_to_tensor(mean, dtype=tf.float32)
        logstd = tf.convert_to_tensor(logstd, dtype=tf.float32)
        n_samples = tf.convert_to_tensor(n_samples, dtype=tf.int32)
        try:
            static_base_shape = mean.get_shape().merge_with(logstd.get_shape())
        except ValueError:
            raise ValueError(
                "mean and logstd must have the same shape (%s vs %s)"
                % (mean.get_shape(), logstd.get_shape()))
        static_base_shape = static_base_shape.as_list()
        _assert_shape_match = tf.assert_equal(
            tf.shape(mean), tf.shape(logstd),
            message="mean and logstd must have the same shape")
        with tf.control_dependencies([_assert_shape_match]):
            base_shape = tf.shape(mean)
        std = tf.exp(logstd)
        if check_numerics:
            with tf.control_dependencies([tf.check_numerics(std, "std")]):
                std = tf.identity(std)
        if not reparameterized:
            mean = tf.stop_gradient(mean)
            std = tf.stop_gradient(std)
        if sample_dim is None:
            _assert_one_sample = tf.assert_equal(
                n_samples, 1,
                message="n_samples must be 1 when sample_dim is None")
            with tf.control_dependencies([_assert_one_sample]):
                shape = tf.identity(base_shape)
                static_shape = static_base_shape
        else:
            sample_dim = tf.convert_to_tensor(sample_dim, tf.int32)
            _assert_positive_dim = tf.assert_greater_equal(
                sample_dim, 0, message="Only support non-negative sample_dim")
            with tf.control_dependencies([_assert_positive_dim]):
                sample_dim = tf.identity(sample_dim)
            shape = tf.concat([base_shape[:sample_dim],
                               tf.stack([n_samples]),
                               base_shape[sample_dim:]], 0)
            if static_sample_dim is not None:
                static_shape = static_base_shape[:static_sample_dim] + \
                    [static_n_samples] + static_base_shape[static_sample_dim:]
            else:
                static_shape = [None] * (len(static_base_shape) + 1)
            mean = tf.expand_dims(mean, sample_dim)
            std = tf.expand_dims(std, sample_dim)
        samples = tf.random_normal(shape) * std + mean
        samples.set_shape(static_shape)
        return samples

    @add_name_scope
    def logpdf(self,
               x,
               mean=0.,
               logstd=0.,
               sample_dim=None,
               check_numerics=True):
        """
        Log probability density function of Normal distribution.

        :param x: A Tensor, python value, or numpy array. The value at which to
            evaluate the log density function. Can be broadcast to match
            `mean` and `logstd`.
        :param mean: A Tensor, python value, or numpy array. The mean of the
            Normal. Can be broadcast to match `x` and `logstd`.
        :param logstd: A Tensor, python value, or numpy array. The log standard
            deviation of the Normal. Can be broadcast to match `x` and `mean`.
        :param sample_dim: A Tensor scalar, int, or None. The sample dimension
            which `x` has more than `mean` and `logstd`. If None, `x` is
            supposed to be a one-sample result, which remains the same shape
            with `mean` and `logstd` (after broadcast).
        :param check_numerics: Bool. Whether to check numeric issues.

        :return: A Tensor of the same shape as `x` (after broadcast) with
            function values.
        """
        static_sample_dim = convert_to_int(sample_dim)
        if static_sample_dim and not (static_sample_dim >= 0):
            raise ValueError("Only support non-negative sample_dim ({})".
                             format(static_sample_dim))
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        mean = tf.convert_to_tensor(mean, dtype=tf.float32)
        logstd = tf.convert_to_tensor(logstd, dtype=tf.float32)
        if sample_dim is not None:
            sample_dim = tf.convert_to_tensor(sample_dim, tf.int32)
            _assert_positive_dim = tf.assert_greater_equal(
                sample_dim, 0, message="Only support non-negative sample_dim")
            with tf.control_dependencies([_assert_positive_dim]):
                sample_dim = tf.identity(sample_dim)
            mean_shape = mean.get_shape().as_list()
            logstd_shape = logstd.get_shape().as_list()
            mean = tf.expand_dims(mean, sample_dim)
            logstd = tf.expand_dims(logstd, sample_dim)
            if static_sample_dim is not None:
                mean.set_shape(mean_shape[:static_sample_dim] + [1] +
                               mean_shape[static_sample_dim:])
                logstd.set_shape(logstd_shape[:static_sample_dim] + [1] +
                                 logstd_shape[static_sample_dim:])
        c = -0.5 * np.log(2 * np.pi)
        acc = tf.exp(-2 * logstd)
        if check_numerics:
            with tf.control_dependencies([tf.check_numerics(acc, "acc")]):
                acc = tf.identity(acc)
        return c - logstd - 0.5 * acc * tf.square(x - mean)


class Logistic(object):
    """
    Class of Logistic distribution.
    """

    def __init__(self):
        pass

    @add_name_scope
    def rvs(self,
            mean=0.,
            logstd=0.,
            sample_dim=None,
            n_samples=1,
            reparameterized=False):
        raise NotImplementedError()

    @add_name_scope
    def cdf(self, x, mean=0., logstd=0., sample_dim=None, check_numerics=True):
        """
        Cumulative distribution function of Logistic distribution.

        :param x: A Tensor, python value, or numpy array. The value at which to
            evaluate the log density function. Can be broadcast to match
            `mean` and `logstd`.
        :param mean: A Tensor, python value, or numpy array. The mean of the
            Normal. Can be broadcast to match `x` and `logstd`.
        :param logstd: A Tensor, python value, or numpy array. The log standard
            deviation of the Normal. Can be broadcast to match `x` and `mean`.
        :param sample_dim: A Tensor scalar, int, or None. The sample dimension
            which `x` has more than `mean` and `logstd`. If None, `x` is
            supposed to be a one-sample result, which remains the same shape
            with `mean` and `logstd` (after broadcast).
        :param check_numerics: Bool. Whether to check numeric issues.

        :return: A Tensor of the same shape as `x` (after broadcast) with
            function values.
        """
        static_sample_dim = convert_to_int(sample_dim)
        if static_sample_dim and not (static_sample_dim >= 0):
            raise ValueError("Only support non-negative sample_dim ({})".
                             format(static_sample_dim))
        mean = tf.convert_to_tensor(mean, dtype=tf.float32)
        logstd = tf.convert_to_tensor(logstd, dtype=tf.float32)
        if sample_dim is not None:
            sample_dim = tf.convert_to_tensor(sample_dim, tf.int32)
            _assert_positive_dim = tf.assert_greater_equal(
                sample_dim, 0, message="Only support non-negative sample_dim")
            with tf.control_dependencies([_assert_positive_dim]):
                sample_dim = tf.identity(sample_dim)
            mean_shape = mean.get_shape().as_list()
            logstd_shape = logstd.get_shape().as_list()
            mean = tf.expand_dims(mean, sample_dim)
            logstd = tf.expand_dims(logstd, sample_dim)
            if static_sample_dim is not None:
                mean.set_shape(mean_shape[:static_sample_dim] + [1] +
                               mean_shape[static_sample_dim:])
                logstd.set_shape(logstd_shape[:static_sample_dim] + [1] +
                                 logstd_shape[static_sample_dim:])
        inv_std = tf.exp(-logstd)
        if check_numerics:
            with tf.control_dependencies(
                    [tf.check_numerics(inv_std, "inv_std")]):
                inv_std = tf.identity(inv_std)
        x_sd = (x - mean) * inv_std
        return tf.sigmoid(x_sd)


class Bernoulli(object):
    """
    Class of Bernoulli distribution.
    """

    def __init__(self):
        pass

    @add_name_scope
    def rvs(self, logits, sample_dim=None, n_samples=1):
        """
        Generate independent Bernoulli samples which forms a Tensor.

        :param logits: A Tensor, python value, or numpy array. The unnormalized
            log probabilities of being 1. (:math:`logits=\log \frac{p}{1 - p}`)
        :param sample_dim: A Tensor scalar, int, or None. The sample dimension.
            If None, this means no new sample dimension is created. In this
            case `n_samples` must be set to 1, otherwise an Exception is
            raised.
        :param n_samples: A Tensor scalar or int. Number of samples to
            generate.

        :return: A Tensor. Samples from the Bernoulli distribution.
        """
        static_sample_dim = convert_to_int(sample_dim)
        if static_sample_dim and not (static_sample_dim >= 0):
            raise ValueError("Only support non-negative sample_dim ({})".
                             format(static_sample_dim))
        static_n_samples = convert_to_int(n_samples)
        logits = tf.convert_to_tensor(logits, dtype=tf.float32)
        n_samples = tf.convert_to_tensor(n_samples, dtype=tf.int32)
        p = tf.sigmoid(logits)
        base_shape = tf.shape(p)
        static_base_shape = p.get_shape().as_list()
        if sample_dim is None:
            _assert_one_sample = tf.assert_equal(
                n_samples, 1,
                message="n_samples must be 1 when sample_dim is None")
            with tf.control_dependencies([_assert_one_sample]):
                shape = tf.identity(base_shape)
            static_shape = static_base_shape
        else:
            sample_dim = tf.convert_to_tensor(sample_dim, tf.int32)
            _assert_positive_dim = tf.assert_greater_equal(
                sample_dim, 0, message="Only support non-negative sample_dim")
            with tf.control_dependencies([_assert_positive_dim]):
                sample_dim = tf.identity(sample_dim)
            p = tf.expand_dims(p, sample_dim)
            shape = tf.concat([base_shape[:sample_dim],
                               tf.stack([n_samples]),
                               base_shape[sample_dim:]], 0)
            if static_sample_dim is not None:
                static_shape = static_base_shape[:static_sample_dim] + \
                    [static_n_samples] + static_base_shape[static_sample_dim:]
            else:
                static_shape = [None] * (len(static_base_shape) + 1)
        alpha = tf.random_uniform(shape, minval=0, maxval=1)
        samples = tf.cast(tf.less(alpha, p), dtype=tf.float32)
        samples.set_shape(static_shape)
        return tf.stop_gradient(samples)

    @add_name_scope
    def logpmf(self, x, logits, sample_dim=None):
        """
        Log probability mass function of Bernoulli distribution.

        :param x: A Tensor, python value, or numpy array. The value at which
            to evaluate the log density function.
        :param logits: A Tensor, python value, or numpy array. The unnormalized
            log probabilities of being 1 (:math:`logits=\log \frac{p}{1 - p}`).
            Must be the same shape with `x` except the `sample_dim`.
        :param sample_dim: A Tensor scalar, int, or None. The sample dimension
            which `x` has more than `logits`. If None, `x` is
            supposed to be a one-sample result, which remains the same shape
            with `logits`.

        :return: A Tensor of the same shape as `x` with function values.
        """
        static_sample_dim = convert_to_int(sample_dim)
        if static_sample_dim and not (static_sample_dim >= 0):
            raise ValueError("Only support non-negative sample_dim ({})".
                             format(static_sample_dim))
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        logits = tf.convert_to_tensor(logits, dtype=tf.float32)
        if sample_dim is not None:
            sample_dim = tf.convert_to_tensor(sample_dim, dtype=tf.int32)
            _assert_positive_dim = tf.assert_greater_equal(
                sample_dim, 0, message="Only support non-negative sample_dim")
            with tf.control_dependencies([_assert_positive_dim]):
                sample_dim = tf.identity(sample_dim)
            logits_shape = logits.get_shape().as_list()
            logits = tf.expand_dims(logits, sample_dim)
            if static_sample_dim is not None:
                logits.set_shape(logits_shape[:static_sample_dim] + [1] +
                                 logits_shape[static_sample_dim:])
            multiples = tf.sparse_to_dense([sample_dim], [tf.rank(logits)],
                                           [tf.shape(x)[sample_dim]], 1)
            logits = tf.tile(logits, multiples)
        try:
            x.get_shape().merge_with(logits.get_shape())
        except ValueError:
            raise ValueError(
                "Shapes of x and logits don't match (%s vs %s)"
                % (x.get_shape(), logits.get_shape()))
        _assert_shape_match = tf.assert_equal(
            tf.shape(x), tf.shape(logits),
            message="Shapes of x and logits must match")
        with tf.control_dependencies([_assert_shape_match]):
            logits = tf.identity(logits)
        return -tf.nn.sigmoid_cross_entropy_with_logits(labels=x,
                                                        logits=logits)


class Discrete(object):
    """
    Class of discrete distribution.
    """

    def __init__(self):
        pass

    @add_name_scope
    def rvs(self, logits, sample_dim=None, n_samples=1):
        """
        Generate samples from Discrete distribution which forms a Tensor.

        :param logits: A N-D (N >= 1) Tensor or numpy array of shape
            (..., n_classes).
            Each slice `[i, j,..., k, :]`
            represents the un-normalized log probabilities for all classes.
        :param sample_dim: A Tensor scalar, int, or None. The sample dimension.
            If None, this means no new sample dimension is created. In this
            case `n_samples` must be set to 1, otherwise an Exception is
            raised.
        :param n_samples: A Tensor scalar or int. Number of samples to
            generate.

        :return: A N-D or (N+1)-D Tensor of shape
            (..., [n_samples], ..., n_classes). Each slice is a one-hot vector
            of the sample.
        """
        static_sample_dim = convert_to_int(sample_dim)
        if static_sample_dim and not (static_sample_dim >= 0):
            raise ValueError("Only support non-negative sample_dim ({})".
                             format(static_sample_dim))
        static_n_samples = convert_to_int(n_samples)
        logits = tf.convert_to_tensor(logits, dtype=tf.float32)
        n_samples = tf.convert_to_tensor(n_samples, dtype=tf.int32)
        base_shape = tf.shape(logits)
        static_base_shape = logits.get_shape().as_list()
        depth = base_shape[-1]
        logits_flat = tf.reshape(logits, [-1, depth])
        samples_flat = tf.one_hot(tf.multinomial(logits_flat, n_samples),
                                  depth, dtype=tf.float32)
        if sample_dim is None:
            _assert_one_sample = tf.assert_equal(
                n_samples, 1,
                message="n_samples must be 1 when sample_dim is None")
            with tf.control_dependencies([_assert_one_sample]):
                samples = tf.reshape(samples_flat, base_shape)
        else:
            sample_dim = tf.convert_to_tensor(sample_dim, tf.int32)
            _assert_positive_dim = tf.assert_greater_equal(
                sample_dim, 0, message="Only support non-negative sample_dim")
            with tf.control_dependencies([_assert_positive_dim]):
                sample_dim = tf.identity(sample_dim)
            shape = tf.concat([base_shape[:-1],
                              tf.stack([n_samples, depth])], 0)
            samples = tf.reshape(samples_flat, shape)
            n_dims = tf.rank(samples)
            dims = tf.range(n_dims)
            sample_dim_mask = tf.cast(tf.one_hot(sample_dim, n_dims), tf.bool)
            original_mask = tf.cast(tf.one_hot(n_dims - 2, n_dims), tf.bool)
            sample_dims = tf.ones([n_dims], tf.int32) * sample_dim
            originals = tf.ones([n_dims], tf.int32) * (n_dims - 2)
            perm = tf.where(original_mask, sample_dims, dims)
            perm = tf.where(sample_dim_mask, originals, perm)
            samples = tf.transpose(samples, perm)
            if static_sample_dim is not None:
                static_shape = static_base_shape[:static_sample_dim] + \
                    [static_n_samples] + static_base_shape[static_sample_dim:]
                samples.set_shape(static_shape)
        return samples

    @add_name_scope
    def logpmf(self, x, logits, sample_dim=None):
        """
        Log probability mass function of Discrete distribution.

        :param x: A N-D (N >= 1) Tensor or numpy array of shape
            (..., n_classes).
            The value at which to evaluate the log probability mass function (
            one-hot).
        :param logits: A N-D (N >= 1) Tensor or numpy array of shape
            (..., n_classes).
            Each slice `[i, j,..., k, :]`
            represents the un-normalized log probabilities for all classes.
            Must be the same shape with `x` except the `sample_dim`.
        :param sample_dim: A Tensor scalar, int, or None. The sample dimension
            which `x` has more than `logits`. If None, `x` is
            supposed to be a one-sample result, which remains the same shape
            with `logits`.

        :return: A (N-1)-D Tensor.
        """
        static_sample_dim = convert_to_int(sample_dim)
        if static_sample_dim and not (static_sample_dim >= 0):
            raise ValueError("Only support non-negative sample_dim ({})".
                             format(static_sample_dim))
        x = tf.convert_to_tensor(x, tf.float32)
        logits = tf.convert_to_tensor(logits, tf.float32)
        if sample_dim is not None:
            sample_dim = tf.convert_to_tensor(sample_dim, dtype=tf.int32)
            _assert_positive_dim = tf.assert_greater_equal(
                sample_dim, 0, message="only support non-negative sample_dim")
            _assert_max_dim = tf.assert_less(
                sample_dim, tf.rank(logits),
                message="only support sample_dim less than tf.rank(logits)")
            with tf.control_dependencies([_assert_positive_dim,
                                          _assert_max_dim]):
                sample_dim = tf.identity(sample_dim)
            logits_shape = logits.get_shape().as_list()
            logits = tf.expand_dims(logits, sample_dim)
            if static_sample_dim is not None:
                logits.set_shape(logits_shape[:static_sample_dim] + [1] +
                                 logits_shape[static_sample_dim:])
            multiples = tf.sparse_to_dense([sample_dim], [tf.rank(logits)],
                                           [tf.shape(x)[sample_dim]], 1)
            logits = tf.tile(logits, multiples)
        try:
            x.get_shape().merge_with(logits.get_shape())
        except ValueError:
            raise ValueError(
                "Shapes of x and logits don't match (%s vs %s)"
                % (x.get_shape(), logits.get_shape()))
        _assert_shape_match = tf.assert_equal(
            tf.shape(x), tf.shape(logits),
            message="Shapes of x and logits must match")
        x_shape = x.get_shape().as_list()
        with tf.control_dependencies([_assert_shape_match]):
            x = tf.argmax(x, tf.rank(x) - 1)
        x.set_shape(x_shape[:-1])
        return -tf.nn.sparse_softmax_cross_entropy_with_logits(labels=x,
                                                               logits=logits)


uniform = Uniform()
norm = Normal()
logistic = Logistic()
bernoulli = Bernoulli()
discrete = Discrete()
