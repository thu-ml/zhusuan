#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
from tensorflow.python.training import moving_averages

from zhusuan.variational.base import VariationalObjective


__all__ = [
    'elbo',
    'EvidenceLowerBoundObjective',
]


class EvidenceLowerBoundObjective(VariationalObjective):
    """
    The class that represents the evidence lower bound (ELBO) objective for
    variational inference. An instance of the class can be constructed by
    calling :func:`elbo`::

        # lower_bound is an EvidenceLowerBoundObjective instance
        lower_bound = zs.variational.elbo(log_joint, observed, latent)

    Instances of :class:`EvidenceLowerBoundObjective` are Tensor-like. They
    can be automatically or manually cast into Tensors when fed into Tensorflow
    Operators and doing computation with Tensors, or when the :attr:`tensor`
    property is accessed. It can also be evaluated like a Tensor::

        # evaluate the ELBO
        with tf.Session() as sess:
            print sess.run(lower_bound, feed_dict=...)

    Maximizing the ELBO wrt. variational parameters is equivalent to
    minimizing :math:`KL(q\|p)`, i.e., the KL-divergence between the
    variational posterior (:math:`q`) and the true posterior (:math:`p`).
    However, this cannot be directly done by calling Tensorflow optimizers on
    the :class:`EvidenceLowerBoundObjective` instance because of the outer
    expectation in the true ELBO objective, while our ELBO value at hand is a
    single or a few sample estimates. The correct way for doing this is by
    calling the gradient estimator provided by
    :class:`EvidenceLowerBoundObjective`. Currently there are two of them:

    * :meth:`sgvb`: The Stochastic Gradient Variational Bayes (SGVB) estimator,
      also known as "the reparameterization trick", or "path derivative
      estimator".
    * :meth:`reinforce`: The score function estimator with variance reduction,
      also known as "REINFORCE", "NVIL", or "likelihood-ratio estimator".

    Thus the typical code for doing variational inference is like::

        # choose a gradient estimator to return the surrogate cost
        cost = lower_bound.sgvb()
        # or
        # cost = lower_bound.reinforce()

        # optimize the surrogate cost wrt. variational parameters
        optimizer = tf.train.AdamOptimizer(learning_rate)
        infer_op = optimizer.minimize(cost,
                                      var_list=variational_parameters)
        with tf.Session() as sess:
            for _ in range(n_iters):
                _, lb = sess.run([infer_op, lower_bound], feed_dict=...)

    .. note::

        Don't directly optimize the :class:`EvidenceLowerBoundObjective`
        instance wrt. variational parameters, i.e., parameters in
        :math:`q`. Instead a proper gradient estimator should be chosen to
        produce the correct surrogate cost to minimize, as shown in the above
        code snippet.

    On the other hand, the ELBO can be used for maximum likelihood learning
    of model parameters, as it is a lower bound of the marginal log
    likelihood of observed variables. Because the outer expectation in the
    ELBO is not related to model parameters, this time it's fine to directly
    optimize the class instance::

        # optimize wrt. model parameters
        learn_op = optimizer.minimize(-lower_bound,
                                      var_list=model_parameters)
        # or
        # learn_op = optimizer.minimize(cost,
        #                               var_list=model_parameters)
        # both ways are correct

    Or we can do inference and learning jointly by optimize over both
    variational and model parameters::

        # joint inference and learning
        infer_and_learn_op = optimizer.minimize(
            cost, var_list=model_and_variational_parameters)

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

    def __init__(self, meta_bn, observed, latent=None, axis=None,
                 variational=None, allow_default=False):
        self._axis = axis
        super(EvidenceLowerBoundObjective, self).__init__(
            meta_bn,
            observed,
            latent=latent,
            variational=variational,
            allow_default=allow_default)

    def _objective(self):
        lower_bound = self._log_joint_term()
        if self._entropy_term() is not None:
            lower_bound += self._entropy_term()
        if self._axis is not None:
            lower_bound = tf.reduce_mean(lower_bound, self._axis)
        return lower_bound

    def sgvb(self):
        """
        Implements the stochastic gradient variational bayes (SGVB) gradient
        estimator for the ELBO, also known as "reparameterization trick" or
        "path derivative estimator".

        It only works for latent `StochasticTensor` s that can be
        reparameterized (Kingma, 2013). For example,
        :class:`~zhusuan.framework.stochastic.Normal`
        and :class:`~zhusuan.framework.stochastic.Concrete`.

        .. note::

            To use the :meth:`sgvb` estimator, the ``is_reparameterized``
            property of each latent `StochasticTensor` must be True (which is
            the default setting when they are constructed).

        :return: A Tensor. The surrogate cost for Tensorflow optimizers to
            minimize.
        """
        return -self.tensor

    def reinforce(self,
                  variance_reduction=True,
                  baseline=None,
                  decay=0.8):
        """
        Implements the score function gradient estimator for the ELBO, with
        optional variance reduction using moving mean estimate or "baseline".
        Also known as "REINFORCE" (Williams, 1992), "NVIL" (Mnih, 2014),
        and "likelihood-ratio estimator" (Glynn, 1990).

        It works for all types of latent `StochasticTensor` s.

        .. note::

            To use the :meth:`reinforce` estimator, the ``is_reparameterized``
            property of each reparameterizable latent `StochasticTensor` must
            be set False.

        :param variance_reduction: Bool. Whether to use variance reduction.
            By default will subtract the learning signal with a moving mean
            estimation of it. Users can pass an additional customized baseline
            using the baseline argument, in that way the returned will be a
            tuple of costs, the former for the gradient estimator, the latter
            for adapting the baseline.
        :param baseline: A Tensor that can broadcast to match the shape
            returned by `log_joint`. A trainable estimation for the scale of
            the elbo value, which is typically dependent on observed values,
            e.g., a neural network with observed values as inputs. This will be
            additional.
        :param decay: Float. The moving average decay for variance
            normalization.

        :return: A Tensor. The surrogate cost for Tensorflow optimizers to
            minimize.
        """
        l_signal = self._log_joint_term() + self._entropy_term()
        baseline_cost = None

        if variance_reduction:
            if baseline is not None:
                baseline_cost = 0.5 * tf.square(
                    tf.stop_gradient(l_signal) - baseline)
                if self._axis is not None:
                    baseline_cost = tf.reduce_mean(baseline_cost, self._axis)
                l_signal = l_signal - baseline

            # TODO: extend to non-scalar.
            bc = tf.reduce_mean(l_signal)
            # TODO: fix get variable failure for repeated calls.
            moving_mean = tf.get_variable(
                'moving_mean', shape=[],
                initializer=tf.constant_initializer(0.),
                trainable=False)

            update_mean = moving_averages.assign_moving_average(
                moving_mean, bc, decay=decay)
            l_signal = l_signal - moving_mean
            with tf.control_dependencies([update_mean]):
                l_signal = tf.identity(l_signal)

        cost = -self._log_joint_term()
        if self._entropy_term() is not None:
            cost += tf.stop_gradient(l_signal) * self._entropy_term()

        if self._axis is not None:
            cost = tf.reduce_mean(cost, self._axis)

        if baseline_cost is not None:
            return cost, baseline_cost
        else:
            return cost


def elbo(meta_bn, observed, latent=None, axis=None, variational=None,
         allow_default=False):
    """
    The evidence lower bound (ELBO) objective for variational inference. The
    returned value is a :class:`EvidenceLowerBoundObjective` instance.

    See :class:`EvidenceLowerBoundObjective` for examples of usage.

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

    :return: An :class:`EvidenceLowerBoundObjective` instance.
    """
    return EvidenceLowerBoundObjective(
        meta_bn,
        observed,
        latent=latent,
        axis=axis,
        variational=variational,
        allow_default=allow_default)
