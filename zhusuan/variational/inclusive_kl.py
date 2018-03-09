#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import warnings

from zhusuan.variational.base import VariationalObjective


__all__ = [
    'klpq',
    'InclusiveKLObjective',
]


class InclusiveKLObjective(VariationalObjective):
    """
    The class that represents the inclusive KL objective (:math:`KL(p\|q)`,
    i.e., the KL-divergence between the true posterior :math:`p` and the
    variational posterior :math:`q`). This is the opposite direction of
    the one (:math:`KL(q\|p)`, or exclusive KL objective) that induces the
    ELBO objective.

    An instance of the class can be constructed by calling :func:`klpq`::

        # klpq_obj is an InclusiveKLObjective instance
        klpq_obj = zs.variational.klpq(
            log_joint, observed, latent, axis=axis)

    Unlike most :class:`~zhusuan.variational.base.VariationalObjective`
    instances, the instance of :class:`InclusiveKLObjective` cannot be used
    like a Tensor or evaluated, because in general this objective is not
    computable.

    The only thing one could achieve with this objective is purely for
    inference, i.e., optimize it wrt. variational parameters (parameters in
    $q$). The way to perform this is by calling the supported gradient
    estimator and getting the surrogate cost to minimize. Currently there is

    * :meth:`rws`: The self-normalized importance sampling gradient estimator.

    So the typical code for doing variational inference is like::

        # call the gradient estimator to return the surrogate cost
        cost = klpq_obj.rws()

        # optimize the surrogate cost wrt. variational parameters
        optimizer = tf.train.AdamOptimizer(learning_rate)
        infer_op = optimizer.minimize(cost,
                                      var_list=variational_parameters)
        with tf.Session() as sess:
            for _ in range(n_iters):
                _, lb = sess.run([infer_op, lower_bound], feed_dict=...)

    .. note::

        The inclusive KL objective is only a criteria for variational
        inference but not model learning (Optimizing it doesn't do maximum
        likelihood learning like the ELBO objective does). That means, there
        is no reason to optimize the surrogate cost wrt. model parameters.

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
    :param axis: The sample dimension(s) to reduce when computing the
        outer expectation in the objective. If ``None``, no dimension is
        reduced.
    """

    def __init__(self, log_joint, observed, latent, axis=None):
        self._axis = axis
        super(InclusiveKLObjective, self).__init__(log_joint, observed, latent)

    def _objective(self):
        raise NotImplementedError(
            "The inclusive KL objective (klpq) can only be optimized instead "
            "of being evaluated.")

    def rws(self):
        """
        Implements the self-normalized importance sampling gradient estimator
        for variational inference. This was used in the Reweighted Wake-Sleep
        (RWS) algorithm (Bornschein, 2015) to adapt the proposal, or
        variational posterior in the importance weighted objective (See
        :class:`~zhusuan.variational.monte_carlo.ImportanceWeightedObjective`).
        Now this estimator is widely used for neural adaptive proposals in
        importance sampling.

        It works for all types of latent `StochasticTensor` s.

        .. note::

            To use the :meth:`rws` estimator, the ``is_reparameterized``
            property of each reparameterizable latent `StochasticTensor` must
            be set False.

        :return: A Tensor. The surrogate cost for Tensorflow optimizers to
            minimize.
        """
        log_w = self._log_joint_term() + self._entropy_term()
        if self._axis is not None:
            log_w_max = tf.reduce_max(log_w, self._axis, keepdims=True)
            w_u = tf.exp(log_w - log_w_max)
            w_tilde = tf.stop_gradient(
                w_u / tf.reduce_sum(w_u, self._axis, keepdims=True))
            cost = tf.reduce_sum(w_tilde * self._entropy_term(),
                                 self._axis)
        else:
            warnings.warn(
                "rws(): The gradient estimator is using self-normalized "
                "importance sampling, which is heavily biased and inaccurate "
                "when you're using only a single sample (`axis=None`).")
            cost = self._entropy_term()
        return cost


def klpq(log_joint, observed, latent, axis=None):
    """
    The inclusive KL objective for variational inference. The
    returned value is an :class:`InclusiveKLObjective` instance.

    See :class:`InclusiveKLObjective` for examples of usage.

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
    :param axis: The sample dimension(s) to reduce when computing the
        outer expectation in the objective. If ``None``, no dimension is
        reduced.

    :return: An :class:`InclusiveKLObjective` instance.
    """
    return InclusiveKLObjective(log_joint, observed, latent, axis=axis)
