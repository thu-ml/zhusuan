#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
from numpy import inf

from zhusuan.distributions.base import Distribution


__all__ = [
    'Empirical',
    'Implicit',
]


class Empirical(Distribution):
    """
    The class of Empirical distribution. Distribution for any variables,
    which are sampled from an empirical distribution and have no explicit
    density. You can not sample from the distribution or calculate
    probabilities and log-probabilities.
    See :class:`~zhusuan.distributions.base.Distribution` for details.

    :param dtype: The value type of samples from the distribution.
    :param batch_shape: A `TensorShape` describing the `batch_shape` of the
        distribution.
    :param value_shape: A `TensorShape` describing the `value_shape` of the
        distribution.
    :param group_ndims: A 0-D `int32` Tensor representing the number of
        dimensions in `batch_shape` (counted from the end) that are grouped
        into a single event, so that their probabilities are calculated
        together. Default is 0, which means a single value is an event.
        See :class:`~zhusuan.distributions.base.Distribution` for more detailed
        explanation.
    :param is_continuous: A bool or `None`. Whether the distribution is
        continuous or not. If `None`, will consider it continuous only if
        `dtype` is a float type.
    """

    def __init__(self,
                 dtype,
                 batch_shape=None,
                 value_shape=None,
                 group_ndims=0,
                 is_continuous=None,
                 **kwargs):
        dtype = tf.float32 if dtype is None else tf.as_dtype(dtype).base_dtype

        self.explicit_batch_shape = tf.TensorShape(batch_shape)

        self.explicit_value_shape = tf.TensorShape(value_shape)

        if is_continuous is None:
            is_continuous = dtype.is_floating

        super(Empirical, self).__init__(
            dtype=dtype,
            param_dtype=None,
            is_continuous=is_continuous,
            is_reparameterized=False,
            use_path_derivative=False,
            group_ndims=group_ndims,
            **kwargs)

    def _value_shape(self):
        raise NotImplementedError()

    def _get_value_shape(self):
        return self.explicit_value_shape

    def _batch_shape(self):
        raise NotImplementedError()

    def _get_batch_shape(self):
        return self.explicit_batch_shape

    def _sample(self, n_samples):
        raise ValueError("You can not sample from an Empirical distribution.")

    def _log_prob(self, given):
        raise ValueError(
            "An empirical distribution has no probability measure.")

    def _prob(self, given):
        raise ValueError(
            "An empirical distribution has no probability measure.")


class Implicit(Distribution):
    """
    The class of Implicit distribution. The distribution abstracts variables
    whose distribution have no explicit form. A common example of implicit
    variables are the generated samples from a GAN.
    See :class:`~zhusuan.distributions.base.Distribution` for details.

    :param samples: A Tensor.
    :param value_shape: A `TensorShape` describing the `value_shape` of the
        distribution.
    :param group_ndims: A 0-D `int32` Tensor representing the number of
        dimensions in `batch_shape` (counted from the end) that are grouped
        into a single event, so that their probabilities are calculated
        together. Default is 0, which means a single value is an event.
        See :class:`~zhusuan.distributions.base.Distribution` for more detailed
        explanation.
    """

    def __init__(self,
                 samples,
                 value_shape=None,
                 group_ndims=0,
                 **kwargs):
        self.samples = tf.convert_to_tensor(samples)

        self.explicit_value_shape = tf.TensorShape(value_shape)

        super(Implicit, self).__init__(
            dtype=samples.dtype,
            param_dtype=samples.dtype,
            is_continuous=samples.dtype.is_floating,
            is_reparameterized=False,
            use_path_derivative=False,
            group_ndims=group_ndims,
            **kwargs)

    def _value_shape(self):
        raise NotImplementedError()

    def _get_value_shape(self):
        return self.explicit_value_shape

    def _batch_shape(self):
        raise NotImplementedError()

    def _get_batch_shape(self):
        if (not self.samples.get_shape()) or (not self.explicit_value_shape):
            return tf.TensorShape(None)
        else:
            d = self.explicit_value_shape.ndims
            if d == 0:
                return self.samples.get_shape()
            else:
                return self.samples.get_shape()[:-d]

    def _sample(self, n_samples=None):
        if n_samples is not None and n_samples != 1:
            raise ValueError(
                "Implicit distribution does not accept `n_samples` argument.")
        return tf.expand_dims(self.samples, 0)

    def _log_prob(self, given):
        return tf.log(self.prob(given))

    def _prob(self, given):
        prob = tf.equal(given, self.samples)
        if self.is_continuous:
            prob = tf.cast(prob, self.param_dtype)
            inf_dtype = tf.cast(inf, self.param_dtype)
            return (2 * prob - 1) * inf_dtype
        else:
            return tf.cast(prob, tf.float32)
