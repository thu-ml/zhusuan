#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

from zhusuan.utils import log_mean_exp
from zhusuan.variational.base import VariationalObjective


__all__ = [
    'importance_weighted_objective',
    'iw_objective',
    'ImportanceWeightedObjective',
]


class ImportanceWeightedObjective(VariationalObjective):
    """
    The class that represents the importance weighted objective for
    variational inference (Burda, 2015). An instance of the class can be
    constructed by calling :func:`importance_weighted_objective`::

        # lower_bound is an ImportanceWeightedObjective instance
        lower_bound = zs.variational.importance_weighted_objective(
            log_joint, observed, latent, axis=axis)

    Instances of :class:`ImportanceWeightedObjective` are Tensor-like. They
    can be automatically or manually cast into Tensors when fed into Tensorflow
    Operators and doing computation with Tensors, or when the :attr:`tensor`
    property is accessed. It can also be evaluated like a Tensor::

        # evaluate the objective
        with tf.Session() as sess:
            print sess.run(lower_bound, feed_dict=...)

    The objective computes the same importance-sampling based estimate
    of the marginal log likelihood of observed variables as
    :meth:`~zhusuan.evaluation.is_loglikelihood`. The difference is that the
    estimate now serves as a variational objective, since it is also a lower
    bound of the marginal log likelihood (as long as the number of samples is
    finite). The variational posterior here is in fact the proposal. As a
    variational objective, :class:`ImportanceWeightedObjective` provides two
    gradient estimators for the variational (proposal) parameters:

    * :meth:`sgvb`: The Stochastic Gradient Variational Bayes (SGVB) estimator,
      also known as "the reparameterization trick", or "path derivative
      estimator".
    * :meth:`vimco`: The multi-sample score function estimator with variance
      reduction, also known as "VIMCO".

    The typical code for joint inference and learning is like::

        # choose a gradient estimator to return the surrogate cost
        cost = lower_bound.sgvb()
        # or
        # cost = lower_bound.vimco()

        # optimize the surrogate cost wrt. model and variational
        # parameters
        optimizer = tf.train.AdamOptimizer(learning_rate)
        infer_and_learn_op = optimizer.minimize(
            cost, var_list=model_and_variational_parameters)
        with tf.Session() as sess:
            for _ in range(n_iters):
                _, lb = sess.run([infer_op, lower_bound], feed_dict=...)

    .. note::

        Don't directly optimize the :class:`ImportanceWeightedObjective`
        instance wrt. to variational parameters, i.e., parameters in
        :math:`q`. Instead a proper gradient estimator should be chosen to
        produce the correct surrogate cost to minimize, as shown in the above
        code snippet.

    Because the outer expectation in the objective is not related to model
    parameters, it's fine to directly optimize the class instance wrt. model
    parameters::

        # optimize wrt. model parameters
        learn_op = optimizer.minimize(-lower_bound,
                                      var_list=model_parameters)
        # or
        # learn_op = optimizer.minimize(cost, var_list=model_parameters)
        # both ways are correct

    The above provides a way for users to combine the importance weighted
    objective with different methods of adapting proposals (:math:`q`). In
    this situation the true posterior is a good choice, which indicates that
    any variational objectives can be used for the adaptation. Specially,
    when the :func:`~zhusuan.variational.inclusive_kl.klpq` objective is
    chosen, this reproduces the Reweighted Wake-Sleep algorithm
    (Bornschein, 2015) for learning deep generative models.

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
        if axis is None:
            raise ValueError(
                "ImportanceWeightedObjective is a multi-sample objective, "
                "the `axis` argument must be specified.")
        self._axis = axis
        super(ImportanceWeightedObjective, self).__init__(
            log_joint, observed, latent)

    def _objective(self):
        log_w = self._log_joint_term() + self._entropy_term()
        if self._axis is not None:
            return log_mean_exp(log_w, self._axis)
        return log_w

    def sgvb(self):
        """
        Implements the stochastic gradient variational bayes (SGVB) gradient
        estimator for the objective, also known as "reparameterization trick"
        or "path derivative estimator". It was first used for importance
        weighted objectives in (Burda, 2015), where it's named "IWAE".

        It only works for latent `StochasticTensor` s that can be
        reparameterized (Kingma, 2013). For example,
        :class:`~zhusuan.model.stochastic.Normal`
        and :class:`~zhusuan.model.stochastic.Concrete`.

        .. note::

            To use the :meth:`sgvb` estimator, the ``is_reparameterized``
            property of each latent `StochasticTensor` must be True (which is
            the default setting when they are constructed).

        :return: A Tensor. The surrogate cost for Tensorflow optimizers to
            minimize.
        """
        return -self.tensor

    def vimco(self):
        """
        Implements the multi-sample score function gradient estimator for
        the objective, also known as "VIMCO", which is named
        by authors of the original paper (Minh, 2016).

        It works for all kinds of latent `StochasticTensor` s.

        .. note::

            To use the :meth:`vimco` estimator, the ``is_reparameterized``
            property of each reparameterizable latent `StochasticTensor` must
            be set False.

        :return: A Tensor. The surrogate cost for Tensorflow optimizers to
            minimize.
        """
        log_w = self._log_joint_term() + self._entropy_term()
        l_signal = log_w

        # check size along the sample axis
        err_msg = "VIMCO is a multi-sample gradient estimator, size along " \
                  "`axis` in the objective should be larger than 1."
        if l_signal.get_shape()[self._axis:self._axis + 1].is_fully_defined():
            if l_signal.get_shape()[self._axis].value < 2:
                raise ValueError(err_msg)
        _assert_size_along_axis = tf.assert_greater_equal(
            tf.shape(l_signal)[self._axis], 2, message=err_msg)
        with tf.control_dependencies([_assert_size_along_axis]):
            l_signal = tf.identity(l_signal)

        # compute variance reduction term
        mean_except_signal = (
            tf.reduce_sum(l_signal, self._axis, keepdims=True) - l_signal
        ) / tf.to_float(tf.shape(l_signal)[self._axis] - 1)
        x, sub_x = tf.to_float(l_signal), tf.to_float(mean_except_signal)

        n_dim = tf.rank(x)
        axis_dim_mask = tf.cast(tf.one_hot(self._axis, n_dim), tf.bool)
        original_mask = tf.cast(tf.one_hot(n_dim - 1, n_dim), tf.bool)
        axis_dim = tf.ones([n_dim], tf.int32) * self._axis
        originals = tf.ones([n_dim], tf.int32) * (n_dim - 1)
        perm = tf.where(original_mask, axis_dim, tf.range(n_dim))
        perm = tf.where(axis_dim_mask, originals, perm)
        multiples = tf.concat(
            [tf.ones([n_dim], tf.int32), [tf.shape(x)[self._axis]]], 0)

        x = tf.transpose(x, perm=perm)
        sub_x = tf.transpose(sub_x, perm=perm)
        x_ex = tf.tile(tf.expand_dims(x, n_dim), multiples)
        x_ex = x_ex - tf.matrix_diag(x) + tf.matrix_diag(sub_x)
        control_variate = tf.transpose(log_mean_exp(x_ex, n_dim - 1),
                                       perm=perm)

        # variance reduced objective
        l_signal = log_mean_exp(l_signal, self._axis,
                                keepdims=True) - control_variate
        fake_term = tf.reduce_sum(
            -self._entropy_term() * tf.stop_gradient(l_signal), self._axis)
        cost = -fake_term - log_mean_exp(log_w, self._axis)

        return cost


def importance_weighted_objective(log_joint, observed, latent, axis=None):
    """
    The importance weighted objective for variational inference (Burda, 2015).
    The returned value is an :class:`ImportanceWeightedObjective` instance.

    See :class:`ImportanceWeightedObjective` for examples of usage.

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

    :return: An :class:`ImportanceWeightedObjective` instance.
    """
    return ImportanceWeightedObjective(log_joint, observed, latent, axis=axis)


# alias
iw_objective = importance_weighted_objective
