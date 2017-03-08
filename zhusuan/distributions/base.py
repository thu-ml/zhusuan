#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import tensorflow as tf

from zhusuan.utils import add_name_scope


__all__ = [
    'Distribution',
]


class Distribution(object):
    """
    The :class:`Distribution` class is the base class for various probabilistic
    distributions which support batch inputs, generating batches of samples and
    evaluate probabilities at batches of given values.

    The typical input shapes for a `Distribution` is like
    `batch_shape + input_shape`. where `input_shape` represents the shape
    of non-batch input parameter, `batch_shape` represents how many independent
    inputs are fed into the distribution.

    Samples generated are of shape `(n + )batch_shape + value_shape`. For
    `n=1`, the first additional axis is omitted. `value_shape` is the non-batch
    value shape of the distribution. For a univariate distribution,
    `value_shape` is [].

    There are cases where batch of random variables are grouped into a
    single event so that their probabilities can be computed together. This
    is achieved by setting `group_event_ndims` argument, which defaults to 0.
    The last `group_event_ndims` number of axes in `batch_shape` are grouped
    into a single event. For example, a `Normal(..., group_event_ndims=1) will
    set the last axis of `batch_shape` to a single event, i.e. a multivariate
    Normal with identity covariance matrix.

    When evaluating probabilities at given values, the given Tensor can be of
    shape `(... + )batch_shape + value_shape`. The returned Tensor has shape
    `(... + )batch_shape[:-group_event_ndims]`.

    :param dtype: The value type of samples from the distribution.
    :param is_continuous: Whether the distribution is continuous.
    :param is_reparameterized: A bool. Whether the gradients of samples can
        and are allowed to propagate back into inputs, using the
        reparametrization trick from (Kingma, 2013).
    :param group_event_ndims: A 0-D `int32` Tensor representing the number of
        dimensions in `batch_shape` (counted from the end) that are grouped
        into a single event, so that their probabilities are calculated
        together. Default is 0, which means a single value is a event.
        See above for more detailed explanation.
    """

    def __init__(self, dtype, is_continuous, is_reparameterized,
                 group_event_ndims=0):
        self._dtype = dtype
        self._is_continuous = is_continuous
        self._is_reparameterized = is_reparameterized
        if isinstance(group_event_ndims, int):
            if group_event_ndims < 0:
                raise ValueError("group_event_ndims must be non-negative.")
            self._group_event_ndims = group_event_ndims
        else:
            group_event_ndims = tf.convert_to_tensor(
                group_event_ndims, tf.int32)
            _assert_rank_op = tf.assert_rank(
                group_event_ndims, 0,
                message="group_event_ndims should be a scalar (0-D Tensor).")
            _assert_nonnegative_op = tf.assert_greater_equal(
                group_event_ndims, 0,
                message="group_event_ndims must be non-negative.")
            with tf.control_dependencies([_assert_rank_op,
                                          _assert_nonnegative_op]):
                self._group_event_ndims = tf.identity(group_event_ndims)

    @property
    def dtype(self):
        """The sample type of the distribution."""
        return self._dtype

    @property
    def is_continuous(self):
        """Whether the distribution is continuous."""
        return self._is_continuous

    @property
    def is_reparameterized(self):
        """
        Whether the gradients of samples can and are allowed to propagate back
        into inputs, using the reparametrization trick from (Kingma, 2013).
        """
        return self._is_reparameterized

    @property
    def group_event_ndims(self):
        """
        The number of dimensions in `batch_shape` (counted from the end)
        that are grouped into a single event, so that their probabilities are
        calculated together. See `Distribution` for more detailed explanation.
        """
        return self._group_event_ndims

    @property
    def value_shape(self):
        """
        The non-batch value shape of a distribution. For batch inputs, the
        shape of a generated sample is `batch_shape + value_shape`.
        """
        static_value_shape = self.get_value_shape()
        if static_value_shape.is_fully_defined():
            return tf.convert_to_tensor(static_value_shape, dtype=tf.int32)
        return self._value_shape()

    def _value_shape(self):
        """
        Private method for subclasses to rewrite the `value_shape` property.
        """
        raise NotImplementedError()

    def get_value_shape(self):
        """
        Static `value_shape`.

        :return: A `TensorShape` instance.
        """
        return self._get_value_shape()

    def _get_value_shape(self):
        """
        Private method for subclasses to rewrite the `get_value_shape` method.
        """
        raise NotImplementedError()

    @property
    def batch_shape(self):
        """
        The shape showing how many independent inputs (which we call batches)
        are fed into the distribution. For batch inputs, the shape of a
        generated sample is `batch_shape + value_shape`.
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
        if isinstance(n_samples, int):
            samples = self._sample(n_samples)
            if n_samples == 1:
                return tf.squeeze(samples, axis=0)
            return samples
        else:
            n_samples = tf.convert_to_tensor(n_samples, dtype=tf.int32)
            _assert_rank_op = tf.assert_rank(
                n_samples, 0,
                message="n_samples should be a scalar (0-D Tensor).")
            with tf.control_dependencies([_assert_rank_op]):
                samples = self._sample(n_samples)
            return tf.cond(tf.equal(n_samples, 1),
                           lambda: tf.squeeze(samples, axis=0),
                           lambda: samples)

    def _sample(self, n_samples):
        """
        Private method for subclasses to rewrite the `sample` method.
        """
        raise NotImplementedError()

    def _check_input_shape(self, given):
        given = tf.convert_to_tensor(given)
        err_msg = "The given argument should be able to broadcast to " \
                  "match batch_shape + value_shape of the distribution."
        if (given.get_shape() and self.get_batch_shape() and
                self.get_value_shape()):
            static_sample_shape = tf.TensorShape(
                self.get_batch_shape().as_list() +
                self.get_value_shape().as_list())
            try:
                tf.broadcast_static_shape(given.get_shape(),
                                          static_sample_shape)
            except ValueError:
                raise ValueError(
                    err_msg + " ({} vs. {} + {})".format(
                        given.get_shape(), self.get_batch_shape(),
                        self.get_value_shape()))
        return given

    @add_name_scope
    def log_prob(self, given):
        """
        Compute log probability density (mass) function at `given` value.

        :param given: A Tensor. The value at which to evaluate log probability
            density (mass) function. Must be able to broadcast to have a shape
            of `(... + )batch_shape + value_shape`.
        :return: A Tensor of shape `(... + )batch_shape[:-group_event_ndims]`.
        """
        given = self._check_input_shape(given)
        log_p = self._log_prob(given)
        return tf.reduce_sum(log_p, tf.range(-self._group_event_ndims, 0))

    @add_name_scope
    def prob(self, given):
        """
        Compute probability density (mass) function at `given` value.

        :param given: A Tensor. The value at which to evaluate probability
            density (mass) function. Must be able to broadcast to have a shape
            of `(... + )batch_shape + value_shape`.
        :return: A Tensor of shape `(... + )batch_shape[:-group_event_ndims]`.
        """
        given = self._check_input_shape(given)
        p = self._prob(given)
        return tf.reduce_prod(p, tf.range(-self._group_event_ndims, 0))

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
