#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import copy

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
    """
    A lazy-constructed :class:`~zhusuan.framework.bn.BayesianNet`. Conceptually
    it's better to view :class:`MetaBayesianNet` rather than
    :class:`~zhusuan.framework.bn.BayesianNet` as the model because it
    can accept different observations through the :meth:`observe` method.

    The suggested usage is through the :func:`meta_bayesian_net` decorator.

    .. seealso::

        For more information, please refer to :doc:`/tutorials/concepts`.

    :param f: A function that constructs and returns a
        :class:`~zhusuan.framework.bn.BayesianNet`.
    :param args: A list. Ordered arguments that will be passed into `f`.
    :param kwargs: A dictionary. Named arguments that will be passed into `f`.
    :param scope: A string. The scope name passed to tensorflow
        `variable_scope()
        <https://www.tensorflow.org/api_docs/python/tf/variable_scope>`_.
    :param reuse_variables: A bool. Whether to reuse tensorflow
        `Variables <https://www.tensorflow.org/api_docs/python/tf/Variable>`_
        in repeated calls of :meth:`observe`.
    """

    def __init__(self, f, args=None, kwargs=None, scope=None,
                 reuse_variables=False):
        if (scope is not None) and reuse_variables:
            self._f = tf.make_template(scope, f)
        elif reuse_variables:
            raise ValueError("Cannot reuse tensorflow Variables when `scope` "
                             "is not provided.")
        else:
            self._f = f
        self._args = copy.copy(args)
        self._kwargs = copy.copy(kwargs)
        self._scope = scope
        self._reuse_variables = reuse_variables
        self._log_joint = None

    @property
    def log_joint(self):
        """
        The log joint function of this model. Can be overwritten as::

            meta_bn = build_model(...)

            def log_joint(bn):
                return ...

            meta_bn.log_joint = log_joint
        """
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
        """
        Construct a :class:`~zhusuan.framework.bn.BayesianNet` given
        observations.

        :param kwargs: A dictionary that maps from node names to their observed
            values.
        :return: A :class:`~zhusuan.framework.bn.BayesianNet` instance.
        """
        if (self._scope is not None) and (not self._reuse_variables):
            with tf.variable_scope(self._scope):
                return self._run_with_observations(self._f, kwargs)
        else:
            return self._run_with_observations(self._f, kwargs)


def meta_bayesian_net(scope=None, reuse_variables=False):
    """
    Transform a function that builds a
    :class:`~zhusuan.framework.bn.BayesianNet` into returning
    :class:`~zhusuan.framework.meta_bn.MetaBayesianNet`.

    The suggested usage is as a decorator::

        @meta_bayesian_net(scope=..., reuse_variables=True)
        def build_model(...):
            bn = zs.BayesianNet()
            ...
            return bn

    The decorated function will return a :class:`MetaBayesianNet` instance
    instead of a :class:`BayesianNet` instance.

    .. seealso::

        For more details and examples, please refer to
        :doc:`/tutorials/concepts`.

    :param scope: A string. The scope name passed to tensorflow
        `variable_scope()
        <https://www.tensorflow.org/api_docs/python/tf/variable_scope>`_.
    :param reuse_variables: A bool. Whether to reuse tensorflow
        `Variables <https://www.tensorflow.org/api_docs/python/tf/Variable>`_
        in repeated calls of :meth:`MetaBayesianNet.observe`.

    :return: The transformed function.
    """
    def wrapper(f):
        @wraps(f)
        def _wrapped(*args, **kwargs):
            meta_bn = MetaBayesianNet(
                f, args=args, kwargs=kwargs, scope=scope,
                reuse_variables=reuse_variables)
            return meta_bn
        return _wrapped
    return wrapper
