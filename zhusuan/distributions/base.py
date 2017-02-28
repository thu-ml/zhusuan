#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from functools import wraps

import tensorflow as tf

from zhusuan.utils import convert_to_int, add_name_scope, doc_inherit


__all__ = [
    'Distribution',
    'ContinuousDistribution',
    'DiscreteDistribution',
    'UnivariateDistribution',
    'MultivariateDistribution'
]


class Distribution(object):
    """
    The :class:`Distribution` class is the base class for various probabilistic
    distributions which support batch inputs, generating batches of samples and
    evaluate probabilities at batches of given values.

    The typical input shapes for a `Distribution` is like
    `batch_shape + event_shape`. We borrow this terminology from
    `tf.contrib.distributions`, where `event_shape` represents the shape
    of non-batch input parameter, `batch_shape` represents how many independent
    inputs are fed into the distribution.

    Samples generated are of shape `[n + ]batch_shape + event_shape`. For
    `n=1`, the first additional axis is omitted.

    When evaluating probabilities at given values, the given Tensor can be of
    shape `[n + ]batch_shape + event_shape`. The returned Tensor has shape
    `[n + ]batch_shape`.

    :param dtype: The value type of samples from the distribution.
    :param is_continuous: Whether the distribution is continuous.
    """

    def __init__(self, dtype, is_continuous, *args, **kwargs):
        self._dtype = dtype
        self._is_continuous = is_continuous

    @property
    def dtype(self):
        """The sample type of the distribution."""
        return self._dtype

    @property
    def is_continuous(self):
        """Whether the distribution is continuous."""
        return self._is_continuous

    @property
    def event_shape(self):
        """
        The shape of a single sample from the distribution with non-batch
        input. For batch inputs, the shape of a generated sample is
        `batch_shape + event_shape`.
        We borrow this concept from `tf.contrib.distributions`.
        """
        static_event_shape = self.get_event_shape()
        if static_event_shape.is_fully_defined():
            return tf.convert_to_tensor(static_event_shape, dtype=tf.int32)
        return self._event_shape()

    def _event_shape(self):
        """
        Private method for subclasses to rewrite the `event_shape` property.
        """
        raise NotImplementedError()

    def get_event_shape(self):
        """
        Static`event_shape`.

        :return: A `TensorShape` instance.
        """
        return self._get_event_shape()

    def _get_event_shape(self):
        """
        Private method for subclasses to rewrite the `get_event_shape` method.
        """
        raise NotImplementedError()

    @property
    def batch_shape(self):
        """
        The shape showing how many independent inputs (which we call batches)
        are fed into the distribution. For batch inputs, the shape of a
        generated sample is `batch_shape + event_shape`.
        We borrow this concept from `tf.contrib.distributions`.
        """
        static_batch_shape = self.get_batch_shape()
        if static_batch_shape.is_fully_defined():
            return tf.convert_to_tensor(static_batch_shape, dtype=tf.int32)
        return self._batch_shape()

    def _batch_shape(self):
        """
        Private method for subclasses to rewrite the `batch_shape` property.
        """
        raise NotImplementedError()

    def get_batch_shape(self):
        """
        Static `batch_shape`.

        :return: A `TensorShape` instance.
        """
        return self._get_batch_shape()

    def _get_batch_shape(self):
        """
        Private method for subclasses to rewrite the `get_batch_shape` method.
        """
        raise NotImplementedError()

    @add_name_scope
    def sample(self, n_samples=1):
        """
        Return samples from the distribution. For `n_samples` larger than 1,
        the returned Tensor has a new sample dimension with size `n_samples`
        inserted at `axis=0`.

        :param n_samples: A 0-D `int32` Tensor. How many independent samples
            to draw from the distribution.
        :return: A Tensor of samples.
        """
        static_n_samples = convert_to_int(n_samples)
        n_samples = tf.convert_to_tensor(n_samples, dtype=tf.int32)
        samples = self._sample(n_samples)
        if static_n_samples is not None:
            if static_n_samples == 1:
                return tf.squeeze(samples, axis=0)
            return samples
        else:
            return tf.cond(tf.equal(n_samples, 1),
                           lambda: tf.squeeze(samples, axis=0),
                           lambda: samples)

    def _sample(self, n_samples):
        """
        Private method for subclasses to rewrite the `sample` method.
        """
        raise NotImplementedError()

    def _call_by_input_rank(f):
        @wraps(f)
        def _func(*args):
            given = tf.convert_to_tensor(args[1])
            static_given_rank = given.get_shape().ndims
            static_batch_rank = args[0].get_batch_shape().ndims
            static_event_rank = args[0].get_event_shape().ndims
            err_msg = "The 'given' argument should have the same or one " \
                      "more rank than the rank of batch_shape + event_shape"
            if (static_given_rank is not None) and (
                    static_batch_rank is not None) and (
                        static_event_rank is not None):
                static_sample_rank = static_batch_rank + static_event_rank
                if static_given_rank == static_sample_rank:
                    given_1 = tf.expand_dims(given, axis=0)
                    return tf.squeeze(f(args[0], given_1), axis=0)
                elif static_given_rank == static_sample_rank + 1:
                    return f(*args)
                else:
                    raise ValueError(
                        err_msg + " (rank {} vs. rank {} + {})".format(
                            static_given_rank, static_batch_rank,
                            static_event_rank))
            else:
                given_rank = tf.rank(given)
                sample_rank = tf.shape(args[0].batch_shape)[0] + \
                    tf.shape(args[0].event_shape)[0]
                assert_rank_op = tf.Assert(
                    tf.logical_or(tf.equal(given_rank, sample_rank),
                                  tf.equal(given_rank, sample_rank + 1)),
                    [given_rank, sample_rank])
                with tf.control_dependencies([assert_rank_op]):
                    return tf.cond(
                        tf.equal(tf.rank(given), sample_rank),
                        lambda: tf.squeeze(
                            f(args[0], tf.expand_dims(given, 0)), 0),
                        lambda: f(*args))
        return _func

    @add_name_scope
    @_call_by_input_rank
    def log_prob(self, given):
        """
        Compute log probability density (mass) function at `given` value.

        :param given: A Tensor. The value at which to evaluate log probability
            density (mass) function. Must be able to broadcast to have a shape
            of `[n + ]batch_shape + event_shape`.
        :return: A Tensor of shape `[n + ]batch_shape`.
        """
        return self._log_prob(given)

    @add_name_scope
    @_call_by_input_rank
    def prob(self, given):
        """
        Compute probability density (mass) function at `given` value.

        :param given: A Tensor. The value at which to evaluate probability
            density (mass) function.
        :return: A Tensor.
        """
        return self._prob(given)

    def _log_prob(self, given):
        """
        Private method for subclasses to rewrite the `log_prob` method.
        """
        raise NotImplementedError()

    def _prob(self, given):
        """
        Private method for subclasses to rewrite the `prob` method.
        """
        raise NotImplementedError()


class ContinuousDistribution(Distribution):
    """
    Base class of continuous distributions.

    :param dtype: The value type of samples from the distribution.
    :param is_reparameterized: A bool. Whether the gradients of samples can
        and are allowed to propagate back into inputs, using the
        reparametrization trick from (Kingma, 2013).
    """

    def __init__(self, dtype, is_reparameterized, *args, **kwargs):
        self._is_reparameterized = is_reparameterized
        super(ContinuousDistribution, self).__init__(
            dtype, is_continuous=True, *args, **kwargs)

    @property
    def is_reparameterized(self):
        """
        Whether the gradients of samples can and are allowed to propagate back
        into inputs, using the reparametrization trick from (Kingma, 2013).
        """
        return self._is_reparameterized


class DiscreteDistribution(Distribution):
    """
    Base class of discrete distributions.

    :param dtype: The value type of samples from the distribution.
    """

    def __init__(self, dtype, *args, **kwargs):
        super(DiscreteDistribution, self).__init__(
            dtype, is_continuous=False, *args, **kwargs)


class UnivariateDistribution(Distribution):
    """
    Base class of univariate distributions.

    :param dtype: The value type of samples from the distribution.
    :param event_ndims: A 0-D `int32` Tensor representing number of
        dimensions of a single event. Default is 0, which means a univariate
        distribution. This is for common cases where we put a group of
        univariate random variables in a single event, so that their
        probabilities are calculated together. For example, when set to 1,
        the last dimension is treated as a single event, the log probabilities
        should sum over this axis.
    """
    def __init__(self, dtype, event_ndims=0, *args, **kwargs):
        static_event_ndims = convert_to_int(event_ndims)
        if static_event_ndims is not None:
            self._event_ndims = static_event_ndims
        else:
            with tf.control_dependencies([tf.assert_rank(event_ndims, 0)]):
                self._event_ndims = tf.identity(event_ndims)
        super(UnivariateDistribution, self).__init__(dtype, *args, **kwargs)

    @property
    def event_ndims(self):
        """
        The number of dimensions of a single event in a univariate
        distribution. This is for common cases where we put a group of
        univariate random variables in a single event, so that their
        probabilities are calculated together.
        """
        return self._event_ndims

    @property
    @doc_inherit
    def event_shape(self):
        b_shape = super(UnivariateDistribution, self).batch_shape
        return b_shape[(tf.shape(b_shape)[0] - self._event_ndims):]

    @doc_inherit
    def get_event_shape(self):
        if isinstance(self._event_ndims, int):
            static_batch_shape = super(UnivariateDistribution,
                                       self).get_batch_shape()
            if static_batch_shape:
                return static_batch_shape[(static_batch_shape.ndims -
                                           self._event_ndims):]
        return tf.TensorShape(None)

    def _event_shape(self):
        return tf.constant([], dtype=tf.int32)

    def _get_event_shape(self):
        return tf.constant([], dtype=tf.int32)

    @property
    @doc_inherit
    def batch_shape(self):
        b_shape = super(UnivariateDistribution, self).batch_shape
        return b_shape[:(tf.shape(b_shape)[0] - self._event_ndims)]

    @doc_inherit
    def get_batch_shape(self):
        if isinstance(self._event_ndims, int):
            static_batch_shape = super(UnivariateDistribution,
                                       self).get_batch_shape()
            if static_batch_shape:
                return static_batch_shape[:(static_batch_shape.ndims -
                                            self._event_ndims)]
        return tf.TensorShape(None)

    @doc_inherit
    def log_prob(self, given):
        log_p = super(UnivariateDistribution, self).log_prob(given)
        return tf.reduce_sum(log_p, tf.range(-self._event_ndims, 0))

    @doc_inherit
    def prob(self, given):
        p = super(UnivariateDistribution, self).log_prob(given)
        return tf.reduce_prod(p, tf.range(-self.event_ndims, 0))


class MultivariateDistribution(Distribution):
    """
    Base class of multivariate distributions.

    :param dtype: The value type of samples from the distribution.
    """
    def __init__(self, dtype, *args, **kwargs):
        super(MultivariateDistribution, self).__init__(dtype, *args, **kwargs)
