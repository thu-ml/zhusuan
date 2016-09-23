#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np

from .utils import as_tensor, add_name_scope


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
    def rvs(self, loc=0., scale=1., shape=None):
        """
        Generate random independent normal samples which form a tensor of
        shape=size. We don't support array-like loc and scale due to
        limitations of tensorflow.random_normal, that is, this can only
        generates i.i.d samples. Also note that we explicitly forbid gradient
        to propagate through the random number generator here.

        :param loc: A 0-D Tensor or python value. The mean of the Normal.
        :param scale: A 0-D Tensor or python value. The standard deviation
            of the Normal.
        :param shape: A 1-D Tensor or numpy array. The shape of the output
            tensor.

        :return: A Tensor of specified shape filled with random i.i.d. normal
            samples.
        """
        loc = tf.cast(as_tensor(loc), dtype=tf.float32)
        scale = tf.cast(as_tensor(scale), dtype=tf.float32)
        return tf.random_normal(shape, tf.stop_gradient(loc),
                                tf.stop_gradient(scale))

    @add_name_scope
    def logpdf(self, x, loc=0., scale=1., eps=1e-8):
        """
        Log probability density function of Normal distribution.

        :param x: A Tensor. The value at which to evaluate the log density
            function.
        :param loc: A Tensor or numpy array. The mean of the Normal.
        :param scale: A Tensor or numpy array. The standard deviation of
            the Normal.

        :return: A Tensor of the same shape as `x` (after broadcast) with
            function values.
        """
        x = tf.cast(as_tensor(x), dtype=tf.float32)
        loc = tf.cast(as_tensor(loc), dtype=tf.float32)
        scale = tf.cast(as_tensor(scale), dtype=tf.float32)
        c = -0.5 * np.log(2 * np.pi)
        scale += eps
        return c - tf.log(scale) - tf.square(x - loc) / (2 * tf.square(scale))


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
        scale += 1e-8
        x_sd = (x - loc) / scale
        return tf.nn.sigmoid(x_sd)


class Bernoulli:
    """
    Class of Bernoulli distribution.
    """
    def __init__(self):
        pass

    @add_name_scope
    def rvs(self, p, size=None):
        """
        Not implemented for now due to absence of Bernoulli in Tensorflow.

        :param p: A Tensor or numpy array. The probability of 1. Can be
            broadcast to size given.
        :param size: A 1-D Tensor or numpy array. The shape of the output
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
        x = tf.cast(as_tensor(x), dtype=tf.float32)
        p = tf.cast(as_tensor(p), dtype=tf.float32)
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

        :param p: A 2-D Tensor or numpy array of shape (n_samples, n_classes).
            Each line is the probability of all classes. (Not required to be
            normalized).
        :param eps: Float. Small value used to avoid NaNs.

        :return: A 2-D Tensor of shape (n_samples, n_classes). Each line is
            a one-hot vector of the sample.
        """
        p = tf.cast(as_tensor(p), dtype=tf.float32)
        tf.assert_rank(p, 2)
        p += eps
        ret = tf.one_hot(
            tf.squeeze(tf.multinomial(tf.stop_gradient(tf.log(p)), 1), [1]),
            tf.shape(p)[1])
        ret.set_shape(p.get_shape())
        return ret

    @add_name_scope
    def logpdf(self, x, p, eps=1e-8):
        """
        Log probability density function of Bernoulli distribution.

        :param x: A 2-D Tensor or numpy array of shape (n_samples, n_classes).
            The value at which to evaluate the log density function (one-hot).
        :param p: A 2-D Tensor or numpy array of shape (n_samples, n_classes).
            Each line is the probability of all classes. (Not required to be
            normalized).
        :param eps: Float. Small value used to avoid NaNs by clipping p in
            range (eps, 1).

        :return: A 1-D Tensor of shape (n_samples,).
        """
        x = tf.cast(as_tensor(x), dtype=tf.float32)
        p = tf.cast(as_tensor(p), dtype=tf.float32)
        tf.assert_rank(x, 2)
        tf.assert_rank(p, 2)
        p += eps
        p = p / tf.reduce_sum(p, 1, keep_dims=True)
        # p = tf.clip_by_value(p, eps, 1.)
        return tf.reduce_sum(x * tf.log(p), 1)


norm = Normal()
logistic = Logistic()
bernoulli = Bernoulli()
discrete = Discrete()
