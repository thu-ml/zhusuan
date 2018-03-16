#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
from functools import wraps

from zhusuan.framework.utils import Context


__all__ = [
    'MetaBayesianNet',
    'meta_bayesian_net',
]


class Local(Context):
    def __getattr__(self, item):
        return self.__dict__.get(item, None)

    def __setattr__(self, key, value):
        self.__dict__[key] = value



class MetaBayesianNet(object):
    def __init__(self, f, args=None, kwargs=None, scope=None,
                 reuse_variables=False):
        if (scope is not None) and reuse_variables:
            self._f = tf.make_template(scope, f)
        elif reuse_variables:
            raise ValueError("Cannot reuse tensorflow Variables when `scope` "
                             "is not provided.")
        else:
            self._f = f
        # TODO: Whether to copy?
        # TODO: make args and kwargs changeable after construction.
        self._args = args
        self._kwargs = kwargs
        self._scope = scope
        self._reuse_variables = reuse_variables
        self._log_joint = None

    @property
    def log_joint(self):
        return self._log_joint

    @log_joint.setter
    def log_joint(self, value):
        self._log_joint = value

    def _run_with_observations(self, func, observations):
        with Local() as local_cxt:
            local_cxt.observations = observations
            local_cxt.meta_bn = self
            return func(*self._args, **self._kwargs)

    def observe(self, **kwargs):
        print("observe:", kwargs)
        if (self._scope is not None) and (not self._reuse_variables):
            with tf.variable_scope(self._scope):
                return self._run_with_observations(self._f, kwargs)
        else:
            return self._run_with_observations(self._f, kwargs)


def meta_bayesian_net(scope=None, reuse_variables=False):
    def wrapper(f):
        @wraps(f)
        def _wrapped(*args, **kwargs):
            meta_bn = MetaBayesianNet(
                f, args=args, kwargs=kwargs, scope=scope,
                reuse_variables=reuse_variables)
            return meta_bn
        return _wrapped
    return wrapper
