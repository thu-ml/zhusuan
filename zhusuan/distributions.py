#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np

from .utils import add_name_scope


__all__ = [
    'norm',
    'logistic',
    'bernoulli',
    'discrete',
]


class Normal:
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
        # TODO: static shape inference
        mean = tf.convert_to_tensor(mean, dtype=tf.float32)
        logstd = tf.convert_to_tensor(logstd, dtype=tf.float32)
        n_samples = tf.convert_to_tensor(n_samples, dtype=tf.int32)
        try:
            mean.get_shape().merge_with(logstd.get_shape())
        except ValueError:
            raise ValueError(
                "mean and logstd must have the same shape (%s vs %s)"
                % (mean.get_shape(), logstd.get_shape()))
        _assert_shape_match = tf.Assert(
            tf.reduce_all(tf.equal(tf.shape(mean), tf.shape(logstd))),
            [tf.shape(mean), tf.shape(logstd)])
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
            _assert_one_sample = tf.Assert(tf.equal(n_samples, 1), [n_samples])
            with tf.control_dependencies([_assert_one_sample]):
                shape = tf.identity(base_shape)
        else:
            # TODO: support negative index
            _assert_positive_dim = tf.Assert(tf.greater_equal(sample_dim, 0),
                                             [sample_dim])
            with tf.control_dependencies([_assert_positive_dim]):
                sample_dim = tf.convert_to_tensor(sample_dim, tf.int32)
            shape = tf.concat(0, [base_shape[:sample_dim],
                                  tf.pack([n_samples]),
                                  base_shape[sample_dim:]])
            mean = tf.expand_dims(mean, sample_dim)
            std = tf.expand_dims(std, sample_dim)
        samples = tf.random_normal(shape) * std + mean
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
        # TODO: static shape inference
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        mean = tf.convert_to_tensor(mean, dtype=tf.float32)
        logstd = tf.convert_to_tensor(logstd, dtype=tf.float32)
        if sample_dim is not None:
            _assert_positive_dim = tf.Assert(tf.greater_equal(sample_dim, 0),
                                             [sample_dim])
            with tf.control_dependencies([_assert_positive_dim]):
                sample_dim = tf.convert_to_tensor(sample_dim, tf.int32)
            mean = tf.expand_dims(mean, sample_dim)
            logstd = tf.expand_dims(logstd, sample_dim)
        c = -0.5 * np.log(2 * np.pi)
        acc = tf.exp(-2 * logstd)
        if check_numerics:
            with tf.control_dependencies([tf.check_numerics(acc, "acc")]):
                acc = tf.identity(acc)
        return c - logstd - 0.5 * acc * tf.square(x - mean)


class Logistic:
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
        mean = tf.convert_to_tensor(mean, dtype=tf.float32)
        logstd = tf.convert_to_tensor(logstd, dtype=tf.float32)
        if sample_dim is not None:
            _assert_positive_dim = tf.Assert(tf.greater_equal(sample_dim, 0),
                                             [sample_dim])
            with tf.control_dependencies([_assert_positive_dim]):
                sample_dim = tf.convert_to_tensor(sample_dim, tf.int32)
            mean = tf.expand_dims(mean, sample_dim)
            logstd = tf.expand_dims(logstd, sample_dim)
        inv_std = tf.exp(-logstd)
        if check_numerics:
            with tf.control_dependencies(
                    [tf.check_numerics(inv_std, "inv_std")]):
                inv_std = tf.identity(inv_std)
        x_sd = (x - mean) * inv_std
        return tf.sigmoid(x_sd)


class Bernoulli:
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
        logits = tf.convert_to_tensor(logits, dtype=tf.float32)
        n_samples = tf.convert_to_tensor(n_samples, dtype=tf.int32)
        p = tf.sigmoid(logits)
        base_shape = tf.shape(p)
        if sample_dim is None:
            _assert_one_sample = tf.Assert(tf.equal(n_samples, 1), [n_samples])
            with tf.control_dependencies([_assert_one_sample]):
                shape = tf.identity(base_shape)
        else:
            sample_dim = tf.convert_to_tensor(sample_dim, tf.int32)
            p = tf.expand_dims(p, sample_dim)
            shape = tf.concat(0, [base_shape[:sample_dim],
                                  tf.pack([n_samples]),
                                  base_shape[sample_dim:]])
        alpha = tf.random_uniform(shape, minval=0, maxval=1)
        samples = tf.cast(tf.less(alpha, p), dtype=tf.float32)
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
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        logits = tf.convert_to_tensor(logits, dtype=tf.float32)
        if sample_dim is not None:
            sample_dim = tf.convert_to_tensor(sample_dim, dtype=tf.int32)
            logits = tf.expand_dims(logits, sample_dim)
            multiples = tf.sparse_to_dense([sample_dim], [tf.rank(logits)],
                                           [tf.shape(x)[sample_dim]], 1)
            logits = tf.tile(logits, multiples)
        _assert_shape_match = tf.Assert(
            tf.reduce_all(tf.equal(tf.shape(logits), tf.shape(x))),
            [tf.shape(logits), tf.shape(x)])
        with tf.control_dependencies([_assert_shape_match]):
            logits = tf.identity(logits)
        return -tf.nn.sigmoid_cross_entropy_with_logits(logits, x)


class Discrete:
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
        logits = tf.convert_to_tensor(logits, dtype=tf.float32)
        n_samples = tf.convert_to_tensor(n_samples, dtype=tf.int32)
        base_shape = tf.shape(logits)
        depth = base_shape[-1]
        logits_flat = tf.reshape(logits, [-1, depth])
        samples_flat = tf.one_hot(tf.multinomial(logits_flat, n_samples),
                                  depth)
        if sample_dim is None:
            _assert_one_sample = tf.Assert(tf.equal(n_samples, 1), [n_samples])
            with tf.control_dependencies([_assert_one_sample]):
                samples = tf.reshape(samples_flat, base_shape)
        else:
            sample_dim = tf.convert_to_tensor(sample_dim, tf.int32)
            shape = tf.concat(0, [base_shape[:-1], tf.pack([n_samples]),
                                  depth])
            samples = tf.reshape(samples_flat, shape)
            n_dims = tf.rank(samples)
            dims = tf.range(n_dims)
            sample_dim_mask = tf.cast(tf.one_hot(sample_dim, 5), tf.bool)
            original_mask = tf.cast(tf.one_hot(n_dims - 2, 5), tf.bool)
            sample_dims = tf.ones(n_dims, tf.int32) * sample_dim
            originals = tf.ones(n_dims, tf.int32) * (n_dims - 2)
            perm = tf.select(original_mask, sample_dims, dims)
            perm = tf.select(sample_dim_mask, originals, perm)
            samples = tf.transpose(samples, perm)
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
        x = tf.convert_to_tensor(x, tf.float32)
        logits = tf.convert_to_tensor(logits, tf.float32)
        if sample_dim is not None:
            sample_dim = tf.convert_to_tensor(sample_dim, dtype=tf.int32)
            logits = tf.expand_dims(logits, sample_dim)
            multiples = tf.sparse_to_dense([sample_dim], [tf.rank(logits)],
                                           [tf.shape(x)[sample_dim]], 1)
            logits = tf.tile(logits, multiples)
        _assert_shape_match = tf.Assert(
            tf.reduce_all(tf.equal(tf.shape(logits), tf.shape(x))),
            [tf.shape(logits), tf.shape(x)])
        with tf.control_dependencies([_assert_shape_match]):
            x = tf.argmax(x, tf.rank(x) - 1)
        return tf.nn.sparse_softmax_cross_entropy_with_logits(logits, x)


norm = Normal()
logistic = Logistic()
bernoulli = Bernoulli()
discrete = Discrete()
