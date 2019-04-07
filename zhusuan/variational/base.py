#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import warnings

import six
import tensorflow as tf
from tensorflow.python.client.session import \
    register_session_run_conversion_functions

from zhusuan.framework.bn import StochasticTensor, BayesianNet
from zhusuan.framework.meta_bn import MetaBayesianNet
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

    def __init__(self, meta_bn, observed, latent=None, variational=None,
                 allow_default=False):
        if callable(meta_bn):
            # TODO: raise warning
            self._meta_bn = None
            self._log_joint = meta_bn
        else:
            self._meta_bn = meta_bn

        if (variational is None) == (latent is None):
            raise ValueError(
                "Either a {} `variational` representing "
                "the variational family or a dictionary `latent` "
                "representing the variational inputs should be passed. "
                "It is not allowed that both are specified or both are not."
                .format(BayesianNet))
        elif latent is None:
            if isinstance(variational, BayesianNet):
                self._variational = variational
            else:
                raise TypeError(
                    "`variational` should be a {} instance, got {}."
                    .format(BayesianNet.__name__, repr(variational)))
            v_inputs = [i for i in six.iteritems(self._variational.nodes)
                        if isinstance(i[1], StochasticTensor) and
                        not i[1].is_observed()]
            v_log_probs = [(name, node.cond_log_p) for name, node in v_inputs]
        else:
            # Deprecated styles of passing variational inputs
            warnings.warn(
                "The `latent` argument has been deprecated and will be "
                "removed in the coming version (0.4.1), use the `variational` "
                "argument instead.", DeprecationWarning)
            self._variational = None
            v_names, v_inputs_and_log_probs = zip(*six.iteritems(latent))
            v_inputs = zip(v_names,
                           map(lambda x: x[0], v_inputs_and_log_probs))
            v_log_probs = zip(v_names,
                              map(lambda x: x[1], v_inputs_and_log_probs))

        # TODO: remove v_log_probs
        self._v_inputs = dict(v_inputs)
        self._v_log_probs = dict(v_log_probs)
        # TODO: Whether to copy?
        self._observed = observed
        self._allow_default = allow_default

    def _validate_variational_inputs(self, bn):
        if self._allow_default:
            # This only works for pure inference settings, or parameter
            # learning when the default variational distribution (i.e., the
            # prior) can be optimized with the chosen estimator.
            return
        for node in bn.nodes:
            if isinstance(node, StochasticTensor) and (not node.is_observed()):
                raise ValueError(
                    "Stochastic node '{}' in the model is neither "
                    "observed nor provided with a variational posterior."
                    .format(node.name))

    @property
    def meta_bn(self):
        return self._meta_bn

    @property
    def variational(self):
        return self._variational

    @property
    def bn(self):
        # TODO: cache bn for the same `meta_bn` and `variational`.
        if self._meta_bn:
            if not hasattr(self, "_bn"):
                self._bn = self._meta_bn.observe(
                    **merge_dicts(self._v_inputs, self._observed))
                self._validate_variational_inputs(self._bn)
            return self._bn
        else:
            return None

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
        if not hasattr(self, "_tensor"):
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

    # TODO: remove deprecated features
    def _log_joint_term(self):
        if self._meta_bn:
            return self.bn.log_joint()
        elif not hasattr(self, '_log_joint_cache'):
            self._log_joint_cache = self._log_joint(
                merge_dicts(self._v_inputs, self._observed))
        return self._log_joint_cache

    def _entropy_term(self):
        if not hasattr(self, '_entropy_cache'):
            if len(self._v_log_probs) > 0:
                self._entropy_cache = -sum(six.itervalues(self._v_log_probs))
            else:
                self._entropy_cache = None
        return self._entropy_cache


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
