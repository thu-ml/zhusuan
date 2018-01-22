#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
import warnings

import numpy as np
import tensorflow as tf
from zhusuan.distributions.base import *

from zhusuan.utils import add_name_scope
from zhusuan.distributions.utils import \
        maybe_explicit_broadcast, \
        assert_same_float_dtype, \
        assert_same_float_and_int_dtype, \
        assert_scalar, \
        assert_rank_at_least_one, \
        get_shape_at, \
        open_interval_standard_uniform


__all__ = [
    'ExponentialFamily',
]

class ExponentialFamily(Distribution):
    """
    Providing the support of the exponential family.
    """
    def __init__(self,
                 natural_param,
                 sufficient_stat_func,
                 condition_func_h,
                 normalization_param,
                 dtype,
                 param_dtype,
                 is_continuous,
                 is_reparameterized,
                 use_path_derivative=False,
                 group_ndims=0,
                 check_numerics=False,
                 **kwargs
                 ):
        self._natural_param = natural_param
        self._normalization_param = normalization_param
        self._sufficient_stat_func = sufficient_stat_func
        self._condition_func_h = condition_func_h
        self._check_numerics = check_numerics
        super(ExponentialFamily, self).__init__(
            dtype=dtype,
            param_dtype=param_dtype,
            is_continuous=is_continuous,
            is_reparameterized=is_reparameterized,
            use_path_derivative=use_path_derivative,
            group_ndims=group_ndims,
            **kwargs)

    def truncate(self):
        pass

    @property
    def natural_param(self):
        """The natural parameter of the distribution."""
        return self._natural_param

    @property
    def normalization_param(self):
        """The normalization parameter of the distribution."""
        return self._normalization_param

    @property
    def parameters(self):
        """The parameters of the distribution."""
        return self._natural_param, self._normalization_param

    def _value_shape(self):
        #return tf.constant([], dtype=tf.int32)
        pass

    def _get_value_shape(self):
        #return tf.TensorShape([])
        pass

    def _batch_shape(self):
        #return tf.broadcast_dynamic_shape(tf.shape(self.mean),
        #                                  tf.shape(self.std))
        pass

    def _get_batch_shape(self):
        #return tf.broadcast_static_shape(self.mean.get_shape(),
        #                                 self.std.get_shape())
        pass

    def _sample(self, n_samples):
        pass

    def _log_prob(self, given):
        pass

    def _prob(self, given):
        return tf.exp(self._log_prob(given))