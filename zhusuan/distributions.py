#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np


__all__ = [
    'norm',
    'bernoulli',
]


class Normal:
    """
    Class of Normal distribution.
    """
    def __init__(self):
        pass

    def rvs(self, loc=0., scale=1., size=None):
        """
        Generate random independent normal samples which form a tensor of
        shape=size. We don't support array-like loc and scale due to
        limitations of tensorflow.random_normal, that is, this can only
        generates i.i.d samples. Also note that we explicitly forbid gradient
        to propagate through the random number generator here.

        :param loc: A 0-D Tensor or python value. The mean of the Normal.
        :param scale: A 0-D Tensor or python value. The standard deviation
            of the Normal.
        :param size: A 1-D Tensor or numpy array. The shape of the output
            tensor.

        :return: A Tensor of specified shape filled with random i.i.d. normal
            samples.
        """
        return tf.random_normal(size, tf.stop_gradient(loc),
                                tf.stop_gradient(scale))

    def logpdf(self, x, loc=0., scale=1.):
        """
        Log probability density function of Normal distribution.

        :param x: A Tensor. The value at which to evaluate the log density
            function.
        :param loc: A Tensor or numpy array. The mean of the Normal.
        :param scale: A Tensor or numpy array. The standard deviation of
            the Normal.

        :return: A Tensor of the same shape as x with function values.
        """
        x = tf.cast(x, dtype=tf.float32)
        loc = tf.cast(loc, dtype=tf.float32)
        scale = tf.cast(scale, dtype=tf.float32)
        c = -0.5 * np.log(2 * np.pi)
        return c - tf.log(scale) - (x - loc)**2 / (2 * scale**2)


class Bernoulli():
    """
    Class of Bernoulli distribution.
    """
    def __init__(self):
        pass

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

    def logpdf(self, x, p, eps=1e-6):
        """
        Log probability density function of Bernoulli distribution.

        :param x: A Tensor. The value at which to evaluate the log density
            function.
        :param p: A Tensor or numpy array. The probability of 1. Can be
            broadcast to the size of x.
        :param eps: Float. Small value used to avoid NaNs by clipping p in
            range (eps, 1 - eps). Should be larger than 1e-6. Default to be
            1e-6.
        :return:
        """
        x = tf.cast(x, dtype=tf.float32)
        p = tf.cast(p, dtype=tf.float32)
        p = tf.clip_by_value(p, eps, 1. - eps)
        return x * tf.log(p) + (1. - x) * tf.log(1. - p)


norm = Normal()
bernoulli = Bernoulli()
