#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
from tensorflow.python.client.session import (
    register_session_run_conversion_functions)
import six

from zhusuan import distributions
from zhusuan.utils import TensorArithmeticMixin
from zhusuan.framework.meta_bn import Local, MetaBayesianNet


__all__ = [
    'StochasticTensor',
    'BayesianNet',
]


# TODO: __str__, __repr__ for StochasticTensor

class StochasticTensor(TensorArithmeticMixin):
    def __init__(self, bn, name, dist, observation=None, **kwargs):
        self._bn = bn
        self._name = name
        self._dist = dist
        self._dtype = dist.dtype
        self._n_samples = kwargs.get("n_samples", None)
        if observation is not None:
            print(name, "set obs: {}".format(observation))
            self._observation = self._check_observation(observation)
        else:
            self._observation = None
        super(StochasticTensor, self).__init__()

    def _check_observation(self, observation):
        type_msg = "Incompatible types of {}('{}') and its observation: {}"
        try:
            observation = tf.convert_to_tensor(observation, dtype=self._dtype)
        except ValueError as e:
            raise type(e)(
                type_msg.format(self.__class__.__name__, self._name, e))

        shape_msg = "Incompatible shapes of {}('{}') and its observation: " \
                    "{} vs {}."
        dist_shape = self._dist.get_batch_shape().concatenate(
            self._dist.get_value_shape())
        try:
            tf.broadcast_static_shape(dist_shape, observation.get_shape())
        except ValueError as e:
            raise type(e)(
                shape_msg.format(
                    self.__class__.__name__, self._name, dist_shape,
                    observation.get_shape()))
        return observation

    @property
    def bn(self):
        return self._bn

    @property
    def name(self):
        return self._name

    @property
    def dtype(self):
        return self._dtype

    @property
    def dist(self):
        return self._dist

    def is_observed(self):
        return self._observation is not None

    @property
    def tensor(self):
        if self._observation is not None:
            return self._observation
        elif not hasattr(self, "_samples"):
            print(self._name, "sample")
            self._samples = self._dist.sample(n_samples=self._n_samples)
        return self._samples

    @property
    def shape(self):
        return self.tensor.shape

    def get_shape(self):
        return self.tensor.get_shape()

    @property
    def cond_log_p(self):
        if not hasattr(self, "_cond_log_p"):
            self._cond_log_p = self._dist.log_prob(self.tensor)
        return self._cond_log_p

    @staticmethod
    def _to_tensor(value, dtype=None, name=None, as_ref=False):
        if dtype and not dtype.is_compatible_with(value.dtype):
            raise ValueError("Incompatible type conversion requested to type "
                             "'{}' for variable of type '{}'".
                             format(dtype.name, value.dtype.name))
        if as_ref:
            raise ValueError("{}: Ref type not supported.".format(value))
        return value.tensor


tf.register_tensor_conversion_function(
    StochasticTensor, StochasticTensor._to_tensor)

# bring support for session.run(StochasticTensor), and for using as keys
# in feed_dict.
register_session_run_conversion_functions(
    StochasticTensor,
    fetch_function=lambda t: ([t.tensor], lambda val: val[0]),
    feed_function=lambda t, v: [(t.tensor, v)],
    feed_function_for_partial_run=lambda t: [t.tensor]
)


class _BayesianNet(object):
    def __init__(self):
        self._nodes = {}
        try:
            self._local_cxt = Local.get_context()
        except RuntimeError:
            self._local_cxt = None
        if self._local_cxt:
            self._meta_bn = self._local_cxt.meta_bn
        else:
            self._meta_bn = None

    @property
    def nodes(self):
        return self._nodes

    def _get_observation(self, name):
        if self._local_cxt:
            ret = self._local_cxt.observations.get(name, None)
            print(name, "get obs: {}".format(ret))
            return ret
        return None

    def stochastic(self, name, dist, **kwargs):
        if name in self._nodes:
            raise ValueError(
                "There exists a node with name '{}' in the {}. Names should "
                "be unique.".format(name, BayesianNet.__name__))
        # TODO: check whether `self` is BayesianNet or _BayesianNet
        print(name, "add stochastic node")
        node = StochasticTensor(
            self, name, dist, observation=self._get_observation(name), **kwargs)
        self._nodes[name] = node
        return node

    def deterministic(self, name, input_tensor):
        input_tensor = tf.convert_to_tensor(input_tensor)
        self._nodes[name] = input_tensor
        return input_tensor

    def _check_name_exist(self, name, only_stochastic=False):
        if not isinstance(name, six.string_types):
            raise TypeError(
                "Expected string in `name_or_names`, got {} of type {}."
                .format(repr(name), type(name)))
        if name not in self._nodes:
            raise ValueError("There isn't a node named '{}' in the {}."
                             .format(name, BayesianNet.__name__))
        elif only_stochastic and not isinstance(
                self._nodes[name], StochasticTensor):
            raise ValueError("Node '{}' is deterministic.".format(name))
        return name

    def _check_names_exist(self, name_or_names, only_stochastic=False):
        """
        Check if there are ``StochasticTensor`` s with `name_or_names` in the
        net.

        :param name_or_names: A string or a tuple(list) of strings. Names of
            ``StochasticTensor`` s in the net.
        :param only_stochastic: A bool. Whether to check only in stochastic
            nodes. Default is `False`.

        :return: A tuple of the validated names.
        """
        if isinstance(name_or_names, six.string_types):
            names = (name_or_names,)
        else:
            names = tuple(name_or_names)
        for name in names:
            _ = self._check_name_exist(name, only_stochastic=only_stochastic)
        return names

    def get(self, name_or_names):
        names = self._check_names_exist(name_or_names)
        ret = [self._nodes[name] for name in names]
        if len(ret) == 1:
            ret = ret[0]
        return ret

    def cond_log_prob(self, name_or_names):
        names = self._check_names_exist(name_or_names, only_stochastic=True)
        ret = [self._nodes[name].cond_log_p for name in names]
        if len(ret) == 1:
            ret = ret[0]
        return ret

    def _log_joint(self):
        if (self._meta_bn is None) or (self._meta_bn.log_joint is None):
            ret = sum(node.cond_log_p for node in six.itervalues(self._nodes)
                      if isinstance(node, StochasticTensor))
        elif callable(self._meta_bn.log_joint):
            ret = self._meta_bn.log_joint(self)
        else:
            raise TypeError(
                "{}.log_joint is set to a non-callable instance: {}"
                .format(self._meta_bn.__class__.__name__,
                        repr(self._meta_bn.log_joint)))
        return ret

    def log_joint(self):
        if not hasattr(self, "_log_joint_cache"):
            self._log_joint_cache = self._log_joint()
        return self._log_joint_cache

    def query(self):
        pass

    def __getitem__(self, name):
        name = self._check_name_exist(name)
        return self._nodes[name]

    def __setitem__(self, name, node):
        raise TypeError(
            "{} instance does not support replacement of the existing node. "
            "To achieve this, pass observations of certain nodes when "
            "calling {}.{}".format(
                BayesianNet.__name__, MetaBayesianNet.__name__,
                MetaBayesianNet.observe.__name__))


class BayesianNet(_BayesianNet):
    def normal(self,
               name,
               mean=0.,
               _sentinel=None,
               std=None,
               logstd=None,
               group_ndims=0,
               n_samples=None,
               is_reparameterized=True,
               check_numerics=False,
               **kwargs):
        dist = distributions.Normal(
            mean,
            _sentinel=_sentinel,
            std=std,
            logstd=logstd,
            group_ndims=group_ndims,
            is_reparameterized=is_reparameterized,
            check_numerics=check_numerics,
            **kwargs
        )
        return self.stochastic(name, dist=dist, n_samples=n_samples, **kwargs)

    def bernoulli(self,
                  name,
                  logits,
                  group_ndims=0,
                  n_samples=None,
                  dtype=tf.int32,
                  **kwargs):
        dist = distributions.Bernoulli(
            logits,
            group_ndims=group_ndims,
            dtype=dtype,
            **kwargs
        )
        return self.stochastic(name, dist=dist, n_samples=n_samples, **kwargs)
