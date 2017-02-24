#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
from tensorflow.contrib import distributions

from .base import StochasticTensor


__all__ = [
    'Uniform',
    'Normal',
    'Bernoulli',
    'Categorical',
    'Discrete',
    'OnehotCategorical',
    'OnehotDiscrete',
]


class ContinuousStochasticTensor(StochasticTensor):
    """
    Base class of continuous `StochasticTensor` s.

    :param name: A string. The name of the StochasticTensor. Must be unique in
        the graph.
    :param dtype: The type of the sampled Tensors.
    :param dist: A `tf.contrib.distributions.Distribution` instance.
    :param is_reparameterized: A bool. Whether the gradients of samples are
        allowed to propagate back into inputs.
    """

    def __init__(self, name, dtype, dist, is_reparameterized, *args, **kwargs):
        self._is_reparameterized = is_reparameterized
        super(ContinuousStochasticTensor, self).__init__(
            name, dtype, dist, is_continuous=True, *args, **kwargs)

    @property
    def is_reparameterized(self):
        """
        Whether the gradients of samples are allowed to propagate back into
        inputs.
        """
        return self._is_reparameterized


class DiscreteStochasticTensor(StochasticTensor):
    """
    Base class of discrete `StochasticTensor` s.

    :param name: A string. The name of the StochasticTensor. Must be unique in
        the graph.
    :param dtype: The type of the sampled Tensors.
    :param dist: A `tf.contrib.distributions.Distribution` instance.
    """

    def __init__(self, name, dtype, dist, *args, **kwargs):
        super(DiscreteStochasticTensor, self).__init__(
            name, dtype, dist, is_continuous=False, *args, **kwargs)


class UnivariateStochasticTensor(StochasticTensor):
    """
    Base class of univariate `StochasticTensor` s.

    :param name: A string. The name of the StochasticTensor. Must be unique in
        the graph.
    :param dtype: The type of the sampled Tensors.
    :param dist: A `tf.contrib.distributions.Distribution` instance.
    :param event_axis: A scalar or 1-D Tensor representing dimensions of a
        single sample. Default is None, which means a univariate distribution.
    """
    def __init__(self, name, dtype, dist, event_axis=None, *args, **kwargs):
        if event_axis is not None:
            with tf.control_dependencies(
                    [tf.assert_rank_in(event_axis, [0, 1])]):
                self._event_axis = tf.identity(event_axis)
        else:
            self._event_axis = None
        super(UnivariateStochasticTensor, self).__init__(
            name, dtype, dist, *args, **kwargs)

    @property
    def event_axis(self):
        return self._event_axis

    def log_prob(self, given):
        log_p = super(UnivariateStochasticTensor, self).log_prob(given)
        return tf.reduce_sum(log_p, axis=self._event_axis)

    def prob(self, given):
        p = super(UnivariateStochasticTensor, self).prob(given)
        return tf.reduce_prod(p, axis=self._event_axis)


class MultivariateStochasticTensor(StochasticTensor):
    """
    Base class of multivariate `StochasticTensor` s.

    :param name: A string. The name of the StochasticTensor. Must be unique in
        the graph.
    :param dtype: The type of the sampled Tensors.
    :param dist: A `tf.contrib.distributions.Distribution` instance.
    """
    def __init__(self, name, dtype, dist, *args, **kwargs):
        super(MultivariateStochasticTensor, self).__init__(
            name, dtype, dist, *args, **kwargs)


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

    def __init__(self,
                 name,
                 minval,
                 maxval,
                 event_axis=None,
                 sample_shape=None,
                 is_reparameterized=True,
                 validate_args=True):
        minval = tf.convert_to_tensor(minval)
        maxval = tf.convert_to_tensor(maxval)
        if not is_reparameterized:
            minval = tf.stop_gradient(minval)
            maxval = tf.stop_gradient(maxval)
        dist = distributions.Uniform(minval, maxval,
                                     validate_args=validate_args,
                                     allow_nan_stats=False)
        super(Uniform, self).__init__(
            name=name,
            dtype=tf.float32,
            dist=dist,
            event_axis=event_axis,
            is_reparameterized=is_reparameterized,
            sample_shape=sample_shape)


class Normal(UnivariateStochasticTensor, ContinuousStochasticTensor):
    """
    The class of univariate Normal `StochasticTensor`.

    :param name: A string. The name of the StochasticTensor. Must be unique in
        the graph.
    :param mean: A Tensor. The mean of the Normal distribution. Should be
        broadcastable to match `logstd`.
    :param logstd: A Tensor. The log standard deviation of the Normal
        distribution. Should be broadcastable to match `mean`.
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
    :param check_numerics: Bool. Whether to check numeric issues.
    """

    def __init__(self,
                 name,
                 mean,
                 logstd,
                 event_axis=None,
                 sample_shape=None,
                 is_reparameterized=True,
                 validate_args=True,
                 check_numerics=True):
        mean = tf.convert_to_tensor(mean)
        logstd = tf.convert_to_tensor(logstd)
        if not is_reparameterized:
            mean = tf.stop_gradient(mean)
            logstd = tf.stop_gradient(logstd)
        # TODO: check_numerics for logstd
        dist = distributions.Normal(mean, logstd, validate_args=validate_args,
                                    allow_nan_stats=False)
        super(Normal, self).__init__(
            name=name,
            dtype=tf.float32,
            dist=dist,
            event_axis=event_axis,
            is_reparameterized=is_reparameterized,
            sample_shape=sample_shape)


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
        logits = tf.convert_to_tensor(logits)
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

    A single sample is a `tf.int32` in [0, n_categories).
    """
    def __init__(self,
                 name,
                 logits,
                 event_axis=None,
                 sample_shape=None,
                 validate_args=False):
        logits = tf.convert_to_tensor(logits)
        dist = distributions.Categorical(logits=logits,
                                         validate_args=validate_args,
                                         allow_nan_stats=False)
        super(Categorical, self).__init__(
            name=name,
            dtype=tf.int32,
            dist=dist,
            event_axis=event_axis,
            sample_shape=sample_shape)


class OnehotCategorical(MultivariateStochasticTensor,
                        ContinuousStochasticTensor):
    """
    The class of one hot Categorical StochasticTensor.

    :param logits: A N-D (N >= 1) Tensor of shape (..., n_categories).
        Each slice `[i, j,..., k, :]` represents the un-normalized log
        probabilities for all categories.
    :param sample_shape: A Tensor. The shape of sample dimensions, which
        indicates how many independent samples to generate.
    :param validate_args: A Bool. Whether to validate input with asserts.
        If `validate_args` is `False`, and the inputs are invalid,
        correct behavior is not guaranteed.

    A single sample is a one-hot vector of shape [n_categories].
    """
    def __init__(self,
                 name,
                 logits,
                 event_axis=None,
                 sample_shape=None,
                 validate_args=False):
        logits = tf.convert_to_tensor(logits)
        dist = distributions.Categorical(logits=logits,
                                         validate_args=validate_args,
                                         allow_nan_stats=False)
        super(OnehotCategorical, self).__init__(
            name=name,
            dtype=tf.int32,
            dist=dist,
            event_axis=event_axis,
            sample_shape=sample_shape)

    def


# alias
Discrete = Categorical
OnehotDiscrete = OnehotCategorical
