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
            logstd=1.,
            sample_dim=None,
            n_samples=1,
            reparameterized=False):
        """
        Generate independent normal samples which forms a Tensor.

        :param mean: A Tensor, python value or numpy array. The mean of the
            Normal distribution.
        :param logstd: A Tensor, python value or numpy array. The log
            standard deviation of the Normal distribution. Should have the same
            shape with `mean`.
        :param sample_dim: A Tensor scalar, int or None. The sample dimension.
            If None, this means no new sample dimension is created. In this
            case `n_samples` must be set to 1, otherwise an Exception is
            raised.
        :param n_samples: A Tensor scalar or int. Number of samples to
            generate.
        :param reparameterized: Bool. If True, gradients on samples from this
            Normal distribution are allowed to propagate into inputs in this
            function, using the reparametrization trick from (Kingma, 2013).

        :return: A Tensor. Samples from the Normal distribution.
        """
        # TODO: static shape inference
        mean = tf.convert_to_tensor(mean, dtype=tf.float32)
        logstd = tf.convert_to_tensor(logstd, dtype=tf.float32)
        _assert_shape_match = tf.Assert(
            tf.equal(tf.shape(mean), tf.shape(logstd)),
            [tf.shape(mean), tf.shape(logstd)])
        with tf.control_dependencies([_assert_shape_match]):
            base_shape = tf.shape(mean)
        std = tf.exp(logstd)
        with tf.control_dependencies([tf.check_numerics(std, "std")]):
            std = tf.identity(std)
        if not reparameterized:
            mean = tf.stop_gradient(mean)
            std = tf.stop_gradient(std)
        if sample_dim is None:
            shape = base_shape
        else:
            sample_dim = tf.convert_to_tensor(sample_dim, tf.int32)
            n_samples = tf.convert_to_tensor(n_samples, tf.int32)
            shape = tf.concat(0, [base_shape[:sample_dim],
                                  tf.pack([n_samples]),
                                  base_shape[sample_dim:]])
            mean = tf.expand_dims(mean, sample_dim)
            std = tf.expand_dims(std, sample_dim)
        samples = tf.random_normal(shape) * std + mean
        return samples

    @add_name_scope
    def logpdf(self, x, mean=0., logstd=1., sample_dim=None):
        """
        Log probability density function of Normal distribution.

        :param x: A Tensor, python value or numpy array. The value at which to
            evaluate the log density function. Can be broadcast to match
            `mean` and `logstd`.
        :param mean: A Tensor, python value or numpy array. The mean of the
            Normal. Can be broadcast to match `x` and `logstd`.
        :param logstd: A Tensor, python value or numpy array. The log standard
            deviation of the Normal. Can be broadcast to match `x` and `mean`.
        :param sample_dim: A Tensor scalar, int or None. The sample dimension
            which `x` has more than `mean` and `logstd`. If None, `x` is
            supposed to be a one-sample result, which remains the same shape
            with mean and logstd.

        :return: A Tensor of the same shape as `x` (after broadcast) with
            function values.
        """
        # TODO: static shape inference
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        mean = tf.convert_to_tensor(mean, dtype=tf.float32)
        logstd = tf.convert_to_tensor(logstd, dtype=tf.float32)
        if sample_dim is not None:
            sample_dim = tf.convert_to_tensor(sample_dim, tf.int32)
            mean = tf.expand_dims(mean, sample_dim)
            logstd = tf.expand_dims(logstd, sample_dim)
        with tf.control_dependencies([tf.check_numerics(logstd, "logstd")]):
            std = tf.exp(logstd)
        c = -0.5 * np.log(2 * np.pi)
        return c - logstd - tf.square(x - mean) / (2 * tf.square(std))


class Logistic:
    """
    Class of Logistic distribution.
    """
    def __init__(self):
        pass

    @add_name_scope
    def rvs(self, loc=0., scale=1., size=None):
        raise NotImplementedError()

    @add_name_scope
    def cdf(self, x, loc=0., scale=1., eps=1e-8):
        """
        Cumulative distribution function of Logistic distribution.

        :param x: A Tensor. The value at which to evaluate the cdf function.
        :param loc: A Tensor or numpy array. The location of the Logistic
            distribution.
        :param scale: A Tensor or numpy array. The scale of the Logistic
            distribution.

        :return: A Tensor of the same shape as `x` (after broadcast) with
            function values.
        """
        scale = tf.clip_by_value(scale, eps, np.inf)
        x_sd = (x - loc) / scale
        return tf.nn.sigmoid(x_sd)


class Bernoulli:
    """
    Class of Bernoulli distribution.
    """
    def __init__(self):
        pass

    @add_name_scope
    def rvs(self, p, shape=None):
        """
        Not implemented for now due to absence of Bernoulli in Tensorflow.

        :param p: A Tensor or numpy array. The probability of 1. Can be
            broadcast to size given.
        :param shape: A 1-D Tensor or numpy array. The shape of the output
            tensor.

        :return: A Tensor of specified shape filled with random independent
            normal samples.
        """
        raise NotImplementedError()

    @add_name_scope
    def logpdf(self, x, p, eps=1e-6):
        """
        Log probability density function of Bernoulli distribution.

        :param x: A Tensor or numpy array. The value at which to evaluate the
            log density function.
        :param p: A Tensor or numpy array. The probability of 1. Can be
            broadcast to the size of `x`.
        :param eps: Float. Small value used to avoid NaNs by clipping p in
            range (eps, 1 - eps). Should be larger than 1e-6. Default to be
            1e-6.

        :return: A Tensor of the same shape as `x` (after broadcast) with
            function values.
        """
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        p = tf.convert_to_tensor(p, dtype=tf.float32)
        p = tf.clip_by_value(p, eps, 1. - eps)
        return x * tf.log(p) + (1. - x) * tf.log(1. - p)


class Discrete:
    """
    Class of discrete distribution.
    """
    def __init__(self):
        pass

    @add_name_scope
    def rvs(self, p, eps=1e-8):
        """
        Generate discrete variables.

        :param p: A N-D (N >=2) Tensor or numpy array of shape
            (n_samples_0, n_samples_1,..., n_classes).
            Each slice `[i, j,..., k, :]`
            represents the un-normalized log probabilities for all classes.

        :param eps: Float. Small value used to avoid NaNs.

        :return: A N-D Tensor of shape
            (n_samples_0, n_samples_1,..., n_classes). Each slice is
            a one-hot vector of the sample.
        """
        p = tf.convert_to_tensor(p, dtype=tf.float32)
        with tf.control_dependencies([tf.assert_rank_at_least(p, 2)]):
            p = tf.clip_by_value(p, eps, np.inf)
        p_flat = tf.reshape(p, [-1, tf.shape(p)[-1]])
        ret_flat = tf.one_hot(
            tf.squeeze(tf.multinomial(tf.stop_gradient(tf.log(p_flat)), 1),
                       [1]), tf.shape(p)[-1])
        ret = tf.reshape(ret_flat, tf.shape(p))
        ret.set_shape(p.get_shape())
        return ret

    @add_name_scope
    def logpdf(self, x, p, eps=1e-8):
        """
        Log probability density function of Discrete distribution.

        :param x: A N-D (N >=2) Tensor or numpy array of shape
            (n_samples_0, n_samples_1,..., n_classes).
            The value at which to evaluate the log density function (one-hot).
        :param p: A N-D (N >=2) Tensor or numpy array of shape
            (n_samples_0, n_samples_1,..., n_classes).
            Each slice `[i, j,..., k, :]`
            represents the un-normalized probabilities for all classes.
        :param eps: Float. Small value used to avoid NaNs by clipping p in
            range (eps, 1).

        :return: A (N-1)-D Tensor of shape (n_samples_0, n_samples_1,..., ).
        """
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        p = tf.convert_to_tensor(p, dtype=tf.float32)
        with tf.control_dependencies([tf.assert_rank_at_least(x, 2),
                                      tf.assert_rank_at_least(p, 2)]):
            p = tf.clip_by_value(p, eps, np.inf)
            return tf.reduce_sum(x * (tf.log(p) - tf.log(
                tf.reduce_sum(p, -1, keep_dims=True))), -1)


norm = Normal()
logistic = Logistic()
bernoulli = Bernoulli()
discrete = Discrete()
