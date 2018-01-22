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
                 dtype,
                 param_dtype,
                 is_continuous,
                 is_reparameterized,
                 use_path_derivative=False,
                 group_ndims=0,
                 **kwargs
                 ):
        super(ExponentialFamily, self).__init__(
            dtype=dtype,
            param_dtype=param_dtype,
            is_continuous=is_continuous,
            is_reparameterized=is_reparameterized,
            use_path_derivative=use_path_derivative,
            group_ndims=group_ndims,
            **kwargs)