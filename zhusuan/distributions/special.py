#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
from numpy import inf

from zhusuan.distributions.base import Distribution
from zhusuan.distributions.utils import assert_same_float_and_int_dtype, \
    maybe_explicit_broadcast


__all__ = [
    'Empirical',
    'Implicit',
]


class Empirical(Distribution):
    """
    The class of Empirical distribution.
    See :class:`~zhusuan.distributions.base.Empirical` for details.

    :param batch_shape: A list or tuple describing the `batch_shape` of the distribution.
        The entries of the list can either be int, Dimension or None.
    :param dtype: The value type of samples from the distribution.
    :param value_shape: A list or tuple describing the `value_shape` of the distribution.
        The entries of the list can either be int, Dimension or None.
    :param group_ndims: A 0-D `int32` Tensor representing the number of
        dimensions in `batch_shape` (counted from the end) that are grouped
        into a single event, so that their probabilities are calculated
        together. Default is 0, which means a single value is an event.
        See :class:`~zhusuan.distributions.base.Distribution` for more detailed
        explanation.
    :param is_continuous: Whether the distribution is continuous or not.
        If None will consider it continuous only if `dtype` is a float type.
    """

    def __init__(self, batch_shape, dtype,
                 value_shape=None,
                 group_ndims=0,
                 is_continuous=None,
                 **kwargs):
        self.explicit_batch_shape = tf.TensorShape(batch_shape)

        if dtype is None:
            dtype = tf.float32
        assert_same_float_and_int_dtype([], dtype)

        if value_shape is None:
            self.explicit_value_shape = tf.TensorShape([])
        elif not isinstance(value_shape, (list, tuple)):
            self.explicit_value_shape = tf.TensorShape([value_shape])
        else:
            self.explicit_value_shape = tf.TensorShape(list(value_shape))

        if is_continuous is None:
            is_continuous = dtype.is_floating

        super(Empirical, self).__init__(
            dtype=dtype,
            param_dtype=None,
            is_continuous=is_continuous,
            is_reparameterized=False,
            group_ndims=group_ndims,
            **kwargs)

    def _value_shape(self):
        return tf.convert_to_tensor(self.explicit_value_shape, tf.int32)

    def _get_value_shape(self):
        return self.explicit_value_shape

    def _batch_shape(self):
        return tf.convert_to_tensor(self.explicit_batch_shape, tf.int32)

    def _get_batch_shape(self):
        return self.explicit_batch_shape

    def _sample(self, n_samples):
        raise ValueError("You can not sample from an Empirical distribution.")

    def _log_prob(self, given):
        raise ValueError("An empirical distribution has no log-probability measure.")

    def _prob(self, given):
        raise ValueError("An empirical distribution has no probability measure.")


class Implicit(Distribution):
    """
    The class of Implicit distribution.
    See :class:`~zhusuan.distributions.base.Implicit` for details.
    :param name: A string. The name of the `StochasticTensor`. Must be unique
        in the `BayesianNet` context.
    :param implicit: A N-D (N >= 1) `float` Tensor
    :param value_shape: A list or tuple describing the `value_shape` of the distribution.
        The entries of the list can either be int, Dimension or None.
    :param group_ndims: A 0-D `int32` Tensor representing the number of
        dimensions in `batch_shape` (counted from the end) that are grouped
        into a single event, so that their probabilities are calculated
        together. Default is 0, which means a single value is an event.
        See :class:`~zhusuan.distributions.base.Distribution` for more detailed
        explanation.
    """

    def __init__(self,
                 implicit,
                 value_shape=None,
                 group_ndims=0,
                 **kwargs):
        self.implicit = implicit

        if value_shape is None:
            self.explicit_value_shape = []
        elif not isinstance(value_shape, (list, tuple)):
            self.explicit_value_shape = [value_shape]
        else:
            self.explicit_value_shape = list(value_shape)

        super(Implicit, self).__init__(
            dtype=implicit.dtype,
            param_dtype=implicit.dtype,
            is_continuous=implicit.dtype.is_floating,
            group_ndims=group_ndims,
            is_reparameterized=False,
            **kwargs)

    def _value_shape(self):
        return tf.convert_to_tensor(self.explicit_value_shape, tf.int32)

    def _get_value_shape(self):
        return tf.TensorShape(self.explicit_value_shape)

    def _batch_shape(self):
        d = len(self.explicit_value_shape)
        if d == 0:
            return tf.shape(self.implicit)
        else:
            return tf.shape(self.implicit)[:-d]

    def _get_batch_shape(self):
        if self.implicit.get_shape():
            d = len(self.explicit_value_shape)
            if d == 0:
                return self.implicit.get_shape()
            else:
                return self.implicit.get_shape()[:-d]
        return tf.TensorShape(None)

    def _sample(self, n_samples=None):
        if n_samples is not None and n_samples != 1:
            raise ValueError("ImplicitDistribution does not accept `n_samples` argument.")
        return tf.expand_dims(self.implicit, 0)

    def _log_prob(self, given):
        return tf.log(self.prob(given))

    def _prob(self, given):
        given = tf.cast(given, self.param_dtype)
        given, implicit = maybe_explicit_broadcast(given, self.implicit, 'given', 'implicit')
        prob = tf.cast(tf.equal(given, implicit), tf.float32)
        if self.is_continuous:
            return (2 * prob - 1) * inf
        else:
            return prob
