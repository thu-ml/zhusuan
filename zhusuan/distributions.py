#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np


__all__ = [
    'norm',
]


class Normal:
    """Class of Normal distribution
    """
    def __init__(self):
        pass

    def rvs(self, loc=0., scale=1., size=None):
        """
        Generate random i.i.d. normal samples which form a tensor of shape=size.
        Note that we explicitly forbid gradient to propagate through the random
        number generator here.

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

        :param x: A Tensor. The value at which to evaluate the density function.
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

    def pdf(self):
        pass


norm = Normal()
