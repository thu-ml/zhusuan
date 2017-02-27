#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import tensorflow as tf

from .base import *


class Normal(UnivariateDistribution, ContinuousDistribution):
    """
    The class of univariate Normal distribution.

    :param mean: A Tensor. The mean of the Normal distribution. Should be
        broadcastable to match `logstd`.
    :param logstd: A Tensor. The log standard deviation of the Normal
        distribution. Should be broadcastable to match `mean`.
    :param event_n_dims: A scalar or 1-D Tensor representing dimensions of a
        single sample. Default is None, which means a univariate distribution.
    :param is_reparameterized: A Bool. If True, gradients on samples from this
        distribution are allowed to propagate into inputs, using the
        reparametrization trick from (Kingma, 2013).
    :param check_numerics: Bool. Whether to check numeric issues.
    """

    def __init__(self,
                 mean,
                 logstd,
                 event_n_dims=0,
                 is_reparameterized=True,
                 *args, **kwargs):
        mean = tf.convert_to_tensor(mean)
        logstd = tf.convert_to_tensor(logstd)
        if not is_reparameterized:
            mean = tf.stop_gradient(mean)
            logstd = tf.stop_gradient(logstd)
        # TODO: check_numerics for logstd
        super(Normal, self).__init__(dtype=tf.float32,
                                     event_n_dims=event_n_dims,
                                     is_reparameterized=is_reparameterized)


class Bernoulli(UnivariateStochasticTensor, DiscreteStochasticTensor):
    """
    The class of Bernoulli `StochasticTensor`.

    :param name: A string. The name of the StochasticTensor. Must be unique in
        the graph.
    :param logits: A Tensor. The unnormalized log probabilities of being 1.

        .. math:: \\mathrm{logits}=\\log \\frac{p}{1 - p}

    :param event_axis: A scalar or 1-D Tensor representing dimensions of a
        single sample. Default is None, which means a univariate distribution.
    :param sample_shape: A Tensor. The shape of sample dimensions, which
        indicates how many independent samples to generate.
    :param validate_args: A Bool. Whether to validate input with asserts.
        If `validate_args` is `False`, and the inputs are invalid,
        correct behavior is not guaranteed.
    """
    def __init__(self,
                 name,
                 logits,
                 event_axis=None,
                 sample_shape=None,
                 validate_args=False):
        dist = distributions.Bernoulli(logits=logits,
                                       validate_args=validate_args,
                                       allow_nan_stats=False)
        super(Bernoulli, self).__init__(
            name=name,
            dtype=tf.int32,
            dist=dist,
            event_axis=event_axis,
            sample_shape=sample_shape)


class Categorical(UnivariateStochasticTensor, DiscreteStochasticTensor):
    """
    The class of Categorical StochasticTensor.

    :param logits: A N-D (N >= 1) Tensor of shape (..., n_categories).
        Each slice `[i, j,..., k, :]` represents the un-normalized log
        probabilities for all categories.
    :param event_axis: A scalar or 1-D Tensor representing dimensions of a
        single sample. Default is None, which means a univariate distribution.
    :param sample_shape: A Tensor. The shape of sample dimensions, which
        indicates how many independent samples to generate.
    :param validate_args: A Bool. Whether to validate input with asserts.
        If `validate_args` is `False`, and the inputs are invalid,
        correct behavior is not guaranteed.

    A single sample is a (N-1)-D Tensor with `tf.int32` values in range
    [0, n_categories).
    """
    def __init__(self,
                 name,
                 logits,
                 event_axis=None,
                 sample_shape=None,
                 validate_args=False):
        dist = distributions.Categorical(logits=logits,
                                         validate_args=validate_args,
                                         allow_nan_stats=False)
        super(Categorical, self).__init__(
            name=name,
            dtype=tf.int32,
            dist=dist,
            event_axis=event_axis,
            sample_shape=sample_shape)


class Uniform(UnivariateStochasticTensor, ContinuousStochasticTensor):
    """
    The class of Uniform `StochasticTensor`.

    :param name: A string. The name of the StochasticTensor. Must be unique in
        the graph.
    :param minval: A Tensor. The lower bound on the range of the uniform
        distribution.
    :param maxval: A Tensor. The upper bound on the range of the uniform
        distribution. Should be element-wise  bigger than `minval`.
    :param event_axis: A scalar or 1-D Tensor representing dimensions of a
        single sample. Default is None, which means a univariate distribution.
    :param sample_shape: A Tensor. The shape of sample dimensions, which
        indicates how many independent samples to generate.
    :param is_reparameterized: A Bool. If True, gradients on samples from this
        distribution are allowed to propagate into inputs, using the
        reparametrization trick from (Kingma, 2013).
    :param validate_args: A Bool. Whether to validate input with asserts.
        If `validate_args` is `False`, and the inputs are invalid,
        correct behavior is not guaranteed.
    """

    def __init__(self, name, dtype, is_reparameterized, *args, **kwargs):
        minval = tf.convert_to_tensor(minval)
        maxval = tf.convert_to_tensor(maxval)
        if not is_reparameterized:
            minval = tf.stop_gradient(minval)
            maxval = tf.stop_gradient(maxval)
        dist = distributions.Uniform(minval, maxval,
                                     validate_args=validate_args,
                                     allow_nan_stats=False)
        super(Uniform, self).__init__(name=name, dtype=tf.float32,
                                      is_reparameterized=is_reparameterized,
                                      event_axis=event_axis,
                                      is_reparameterized=is_reparameterized,
                                      sample_shape=sample_shape)
