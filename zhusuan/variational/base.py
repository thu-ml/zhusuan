#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import six
import tensorflow as tf
from tensorflow.python.client.session import \
    register_session_run_conversion_functions

from zhusuan.utils import TensorArithmeticMixin, merge_dicts


__all__ = [
    'VariationalObjective',
]


class VariationalObjective(TensorArithmeticMixin):
    """
    The base class for variational objectives. You never use this class
    directly, but instead instantiate one of its subclasses by calling
    :func:`~zhusuan.variational.exclusive_kl.elbo`,
    :func:`~zhusuan.variational.monte_carlo.importance_weighted_objective`,
    or :func:`~zhusuan.variational.inclusive_kl.klpq`.

    :param log_joint: A function that accepts a dictionary argument of
        ``(string, Tensor)`` pairs, which are mappings from all
        `StochasticTensor` names in the model to their observed values. The
        function should return a Tensor, representing the log joint likelihood
        of the model.
    :param observed: A dictionary of ``(string, Tensor)`` pairs. Mapping from
        names of observed `StochasticTensor` s to their values.
    :param latent: A dictionary of ``(string, (Tensor, Tensor))`` pairs.
        Mapping from names of latent `StochasticTensor` s to their samples and
        log probabilities.
    """

    def __init__(self, log_joint, observed, latent):
        self._log_joint = log_joint
        self._observed = observed
        self._latent = latent
        # TODO: Add input name check by matching them
        latent_k, latent_v = map(list, zip(*six.iteritems(latent)))
        self._latent_outputs = dict(
            zip(latent_k, map(lambda x: x[0], latent_v)))
        self._latent_logpdfs = dict(
            zip(latent_k, map(lambda x: x[1], latent_v)))
        self._joint_obs = merge_dicts(observed, self._latent_outputs)
        try:
            self._dict_key = (log_joint,
                              frozenset(latent_k),
                              frozenset(map(tuple, latent_v)),
                              frozenset(six.iteritems(observed)))
        except TypeError:
            # Unhashable type
            self._dict_key = None

    @classmethod
    def _get_log_p_cache(cls):
        if not hasattr(cls, '_global_log_p'):
            cls._global_log_p = {}
        return cls._global_log_p

    @classmethod
    def _get_log_q_cache(cls):
        if not hasattr(cls, '_global_log_q'):
            cls._global_log_q = {}
        return cls._global_log_q

    def _log_joint_term(self):
        if self._dict_key is not None:
            log_p_cache = VariationalObjective._get_log_p_cache()
            if self._dict_key not in log_p_cache:
                log_p_cache[self._dict_key] = self._log_joint(self._joint_obs)
            return log_p_cache[self._dict_key]
        else:
            if not hasattr(self, '_log_p'):
                self._log_p = self._log_joint(self._joint_obs)
            return self._log_p

    def _entropy_term(self):
        if self._dict_key is not None:
            log_q_cache = VariationalObjective._get_log_q_cache()
            if self._dict_key not in log_q_cache:
                log_q_cache[self._dict_key] = -tf.add_n(
                    list(six.itervalues(self._latent_logpdfs)))
            return log_q_cache[self._dict_key]
        else:
            if not hasattr(self, '_log_q'):
                self._log_q = -tf.add_n(
                    list(six.itervalues(self._latent_logpdfs)))
            return self._log_q

    def _objective(self):
        """
        Private method for subclasses to rewrite the objective value.

        :return: A Tensor representing the value of the objective.
        """
        raise NotImplementedError()

    @property
    def tensor(self):
        """
        Return the Tensor representing the value of the variational objective.

        :return: A Tensor.
        """
        if not hasattr(self, '_tensor'):
            self._tensor = self._objective()
        return self._tensor

    @staticmethod
    def _to_tensor(value, dtype=None, name=None, as_ref=False):
        tensor = value.tensor
        if dtype and not dtype.is_compatible_with(tensor.dtype):
            raise ValueError("Incompatible type conversion requested to type "
                             "'{}' for variable of type '{}'".
                             format(dtype.name, tensor.dtype.name))
        if as_ref:
            raise ValueError("{}: Ref type not supported.".format(value))
        return tensor


tf.register_tensor_conversion_function(
    VariationalObjective, VariationalObjective._to_tensor)

# bring support for session.run(VariationalObjective), and for using as keys
# in feed_dict.
register_session_run_conversion_functions(
    VariationalObjective,
    fetch_function=lambda t: ([t.tensor], lambda val: val[0]),
    feed_function=lambda t, v: [(t.tensor, v)],
    feed_function_for_partial_run=lambda t: [t.tensor]
)
