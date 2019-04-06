#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from copy import copy

import six
from six.moves import zip, map
import tensorflow as tf

from zhusuan.utils import add_name_scope, merge_dicts


__all__ = [
    "HMCInfo",
    "HMC",
]


def random_momentum(shapes, mass):
    return [tf.random_normal(shape=shape) * tf.sqrt(m)
            for shape, m in zip(shapes, mass)]


def velocity(momentum, mass):
    return map(lambda z: z[0] / z[1], zip(momentum, mass))


def hamiltonian(q, p, log_posterior, mass, data_axes):
    potential = -log_posterior(q)
    kinetic = 0.5 * tf.add_n(
        [tf.reduce_sum(tf.square(momentum) / m, axis)
         for momentum, m, axis in zip(p, mass, data_axes)])
    return potential + kinetic, -potential


def leapfrog_integrator(q, p, step_size1, step_size2, grad, mass):
    q = [x + step_size1 * y for x, y in zip(q, velocity(p, mass))]
    # p = p + epsilon / 2 * gradient q
    grads = grad(q)
    p = [x + step_size2 * y for x, y in zip(p, grads)]
    return q, p


def get_acceptance_rate(q, p, new_q, new_p, log_posterior, mass, data_axes):
    old_hamiltonian, old_log_prob = hamiltonian(
        q, p, log_posterior, mass, data_axes)
    new_hamiltonian, new_log_prob = hamiltonian(
        new_q, new_p, log_posterior, mass, data_axes)
    old_log_prob = tf.check_numerics(
        old_log_prob,
        'HMC: old_log_prob has numeric errors! Try better initialization.')
    acceptance_rate = tf.exp(
        tf.minimum(-new_hamiltonian + old_hamiltonian, 0.0))
    is_finite = tf.logical_and(tf.is_finite(acceptance_rate),
                               tf.is_finite(new_log_prob))
    acceptance_rate = tf.where(is_finite, acceptance_rate,
                               tf.zeros_like(acceptance_rate))
    return old_hamiltonian, new_hamiltonian, old_log_prob, new_log_prob, \
        acceptance_rate


class StepsizeTuner:
    def __init__(self, initial_stepsize, adapt_step_size, gamma, t0, kappa,
                 delta):
        with tf.name_scope("StepsizeTuner"):
            self.adapt_step_size = tf.convert_to_tensor(
                adapt_step_size, dtype=tf.bool, name="adapt_step_size")
            self.initial_stepsize = initial_stepsize

            self.gamma = tf.convert_to_tensor(gamma, dtype=tf.float32,
                                              name="gamma")
            self.t0 = tf.convert_to_tensor(t0, dtype=tf.float32, name="t0")
            self.kappa = tf.convert_to_tensor(kappa, dtype=tf.float32,
                                              name="kappa")
            self.delta = tf.convert_to_tensor(delta, dtype=tf.float32,
                                              name="delta")
            self.mu = tf.constant(10 * initial_stepsize, dtype=tf.float32,
                                  name="mu")

            self.step = tf.Variable(0.0, dtype=tf.float32,
                                    name="step", trainable=False)
            self.log_epsilon_bar = tf.Variable(
                0.0, dtype=tf.float32, name="log_epsilon_bar", trainable=False)
            self.h_bar = tf.Variable(0.0, dtype=tf.float32,
                                     name="h_bar", trainable=False)

    @add_name_scope
    def tune(self, acceptance_rate, fresh_start):
        def adapt_stepsize():
            new_step = tf.assign(self.step, (1 - fresh_start) * self.step + 1)
            rate1 = 1.0 / (new_step + self.t0)
            new_h_bar = tf.assign(
                self.h_bar, (1 - fresh_start) * (1 - rate1) * self.h_bar +
                rate1 * (self.delta - acceptance_rate))
            log_epsilon = self.mu - tf.sqrt(new_step) / self.gamma * new_h_bar
            rate = tf.pow(new_step, -self.kappa)
            new_log_epsilon_bar = tf.assign(
                self.log_epsilon_bar,
                rate * log_epsilon + (1 - fresh_start) * (1 - rate) *
                self.log_epsilon_bar)
            with tf.control_dependencies([new_log_epsilon_bar]):
                new_log_epsilon = tf.identity(log_epsilon)

            return tf.exp(new_log_epsilon)

        c = tf.cond(self.adapt_step_size,
                    adapt_stepsize,
                    lambda: tf.exp(self.log_epsilon_bar))

        return c


class ExponentialWeightedMovingVariance:
    def __init__(self, decay, shape, num_chain_dims):
        with tf.name_scope("ExponentialWeightedMovingVariance"):
            self.t = tf.Variable(0.0, name="t", trainable=False)
            # mean, var: (1,...,1 data_dims)
            self.mean = [tf.Variable(tf.zeros(s), name="mean",
                                     trainable=False) for s in shape]
            self.var = [tf.Variable(tf.zeros(s), name="var",
                                    trainable=False) for s in shape]
            self.decay = decay
            self.one = tf.constant(1.0, dtype=tf.float32)
            self.num_chain_dims = num_chain_dims
            self.chain_axes = tf.range(self.num_chain_dims)

    @add_name_scope
    def update(self, x):
        # x: (chain_dims data_dims)
        new_t = tf.assign(self.t, self.t + 1)
        weight = (1 - self.decay) / (1 - tf.pow(self.decay, new_t))
        # incr: (chain_dims data_dims)
        incr = [weight * (q - mean) for q, mean in zip(x, self.mean)]
        # mean: (1,...,1 data_dims)
        update_mean = [mean.assign_add(
            tf.reduce_mean(i, axis=self.chain_axes, keepdims=True))
            for mean, i in zip(self.mean, incr)]
        # var: (1,...,1 data_dims)
        new_var = [
            (1 - weight) * var +
            tf.reduce_mean(i * (q - mean), axis=self.chain_axes,
                           keepdims=True)
            for var, i, q, mean in zip(self.var, incr, x, update_mean)]

        update_var = [tf.assign(var, n_var)
                      for var, n_var in zip(self.var, new_var)]
        return update_var

    def get_precision(self, var_in):
        return [(self.one / var) for var in var_in]

    def get_updated_precision(self, x):
        # Should be called only once
        return self.get_precision(self.update(x))

    def precision(self):
        return self.get_precision(self.var)


class HMCInfo(object):
    """
    Contains information about a sampling iteration by :class:`HMC`. Users
    can get fine control of the sampling process by monitoring these
    statistics.

    .. note::

        Attributes provided in this structure must be fetched together with the
        corresponding sampling operation and should not be fetched anywhere
        else. Otherwise you would get undefined behaviors.

    :param samples: A dictionary of ``(string, Tensor)`` pairs. Samples
        generated by this HMC iteration.
    :param acceptance_rate: A Tensor. The acceptance rate in this iteration.
    :param updated_step_size: A Tensor. The updated step size (by adaptation)
        after this iteration.
    :param init_momentum: A dictionary of ``(string, Tensor)`` pairs. The
        initial momentum for each latent variable in this sampling iteration.
    :param orig_hamiltonian: A Tensor. The original hamiltonian at the
        beginning of the iteration.
    :param hamiltonian: A Tensor. The current hamiltonian at the end of the
        iteration.
    :param orig_log_prob: A Tensor. The log joint probability at the
        beginning position of the iteration.
    :param log_prob: A Tensor. The current log joint probability at the end
        position of the iteration.
    """

    def __init__(self, samples, acceptance_rate, updated_step_size,
                 init_momentum, orig_hamiltonian, hamiltonian, orig_log_prob,
                 log_prob):
        self.samples = samples
        self.acceptance_rate = acceptance_rate
        self.updated_step_size = updated_step_size
        self.init_momentum = init_momentum
        self.orig_hamiltonian = orig_hamiltonian
        self.hamiltonian = hamiltonian
        self.orig_log_prob = orig_log_prob
        self.log_prob = log_prob


class HMC:
    """
    Hamiltonian Monte Carlo (Neal, 2011) with adaptation for stepsize
    (Hoffman, 2014) and mass. The usage is similar with a Tensorflow
    optimizer.

    The :class:`HMC` class supports running multiple MCMC chains in parallel.
    To use the sampler, the user first create a tensorflow `Variable` storing
    the initial sample, whose shape is ``chain axes + data axes``. There
    can be arbitrary number of chain axes followed by arbitrary number of
    data axes. Then the user provides a `log_joint` function which returns
    a tensor of shape ``chain axes``, which is the log joint density for
    each chain. Finally, the user runs the operation returned by
    :meth:`sample`, which updates the sample stored in the variable.

    .. note::

        Currently we do not support invoking the :meth:`sample` method
        multiple times per :class:`HMC` class. Please declare one :class:`HMC`
        class per each invoke of the :meth:`sample` method.

    .. note::

        When the adaptations are on, the sampler is not reversible.
        To guarantee current equilibrium, the user should only turn on
        the adaptations during the burn-in iterations, and turn them off
        when collecting samples. To achieve this, the best practice is to
        set `adapt_step_size` and `adapt_mass` to be placeholders and feed
        different values (True/False) when needed.

    :param step_size: A 0-D `float32` Tensor. Initial step size.
    :param n_leapfrogs: A 0-D `int32` Tensor. Number of leapfrog steps.
    :param adapt_step_size: A `bool` Tensor, if set, indicating whether to
        adapt the step size.
    :param target_acceptance_rate: A 0-D `float32` Tensor. The desired
        acceptance rate for adapting the step size.
    :param gamma: A 0-D `float32` Tensor. Parameter for adapting the step
        size, see (Hoffman, 2014).
    :param t0: A 0-D `float32` Tensor. Parameter for adapting the step size,
        see (Hoffman, 2014).
    :param kappa: A 0-D `float32` Tensor. Parameter for adapting the step
        size, see (Hoffman, 2014).
    :param adapt_mass: A `bool` Tensor, if set, indicating whether to adapt
        the mass, adapt_step_size must be set.
    :param mass_collect_iters: A 0-D `int32` Tensor. The beginning iteration
        to change the mass.
    :param mass_decay: A 0-D `float32` Tensor. The decay of computing
        exponential moving variance.
    """
    def __init__(self, step_size=1., n_leapfrogs=10,
                 adapt_step_size=None, target_acceptance_rate=0.8,
                 gamma=0.05, t0=100, kappa=0.75,
                 adapt_mass=None, mass_collect_iters=10, mass_decay=0.99):
        # TODO: Maintain the variables somewhere else to let the sample be
        # called multiple times
        self.step_size = tf.Variable(step_size, name="step_size",
                                     trainable=False, dtype=tf.float32)
        self.n_leapfrogs = tf.convert_to_tensor(n_leapfrogs, tf.int32,
                                                name="n_leapfrogs")
        self.target_acceptance_rate = tf.convert_to_tensor(
            target_acceptance_rate, tf.float32, name="target_acceptance_rate")
        self.t = tf.Variable(0.0, name="t", trainable=False, dtype=tf.float32)
        self.adapt_step_size = adapt_step_size
        if adapt_step_size is not None:
            self.step_size_tuner = StepsizeTuner(
                step_size, adapt_step_size, gamma, t0, kappa,
                target_acceptance_rate)
        if adapt_mass is not None:
            if adapt_step_size is None:
                raise ValueError('If adapt mass is set, we should also adapt step size')
            self.adapt_mass = tf.convert_to_tensor(
                adapt_mass, dtype=tf.bool, name="adapt_mass")
        else:
            mass_collect_iters = 0
            self.adapt_mass = None
        self.mass_collect_iters = tf.convert_to_tensor(
            mass_collect_iters, tf.int32, name="mass_collect_iters")
        self.mass_decay = tf.convert_to_tensor(
            mass_decay, tf.float32, name="mass_decay")

    @add_name_scope
    def _adapt_mass(self, t, num_chain_dims):
        ewmv = ExponentialWeightedMovingVariance(
            self.mass_decay, self.data_shapes, num_chain_dims)
        new_mass = tf.cond(self.adapt_mass,
                           lambda: ewmv.get_updated_precision(self.q),
                           lambda: ewmv.precision())
        if not isinstance(new_mass, list):
            new_mass = [new_mass]

        # print('New mass is = {}'.format(new_mass))
        # TODO incorrect shape?
        # print('New mass={}'.format(new_mass))
        # print('q={}, NMS={}'.format(self.q[0].get_shape(),
        #                             new_mass[0].get_shape()))
        with tf.control_dependencies(new_mass):
            current_mass = tf.cond(
                tf.less(tf.cast(t, tf.int32), self.mass_collect_iters),
                lambda: [tf.ones(shape) for shape in self.data_shapes],
                lambda: new_mass)
        if not isinstance(current_mass, list):
            current_mass = [current_mass]
        return current_mass

    @add_name_scope
    def _init_step_size(self, q, p, mass, get_gradient, get_log_posterior):
        factor = 1.5

        def loop_cond(step_size, last_acceptance_rate, cond):
            return cond

        def loop_body(step_size, last_acceptance_rate, cond):
            # Calculate acceptance_rate
            new_q, new_p = leapfrog_integrator(
                q, p, tf.constant(0.0), step_size / 2,
                get_gradient, mass)
            new_q, new_p = leapfrog_integrator(
                new_q, new_p, step_size, step_size / 2,
                get_gradient, mass)
            __, _, _, _, acceptance_rate = get_acceptance_rate(
                q, p, new_q, new_p,
                get_log_posterior, mass, self.data_axes)

            acceptance_rate = tf.reduce_mean(acceptance_rate)

            # Change step size and stopping criteria
            new_step_size = tf.cond(
                tf.less(acceptance_rate,
                        self.target_acceptance_rate),
                lambda: step_size * (1.0 / factor),
                lambda: step_size * factor)

            cond = tf.logical_not(tf.logical_xor(
                tf.less(last_acceptance_rate, self.target_acceptance_rate),
                tf.less(acceptance_rate, self.target_acceptance_rate)))
            return [new_step_size, acceptance_rate, cond]

        new_step_size, _, _ = tf.while_loop(
            loop_cond,
            loop_body,
            [self.step_size, tf.constant(1.0), tf.constant(True)]
        )
        return new_step_size

    @add_name_scope
    def _leapfrog(self, q, p, step_size, get_gradient, mass):
        def loop_cond(i, q, p):
            return i < self.n_leapfrogs + 1

        def loop_body(i, q, p):
            step_size1 = tf.cond(i > 0,
                                 lambda: step_size,
                                 lambda: tf.constant(0.0, dtype=tf.float32))

            step_size2 = tf.cond(tf.logical_and(tf.less(i, self.n_leapfrogs),
                                                tf.less(0, i)),
                                 lambda: step_size,
                                 lambda: step_size / 2)

            q, p = leapfrog_integrator(q, p, step_size1, step_size2,
                                       lambda q: get_gradient(q), mass)
            return [i + 1, q, p]

        i = tf.constant(0)
        _, q, p = tf.while_loop(loop_cond,
                                loop_body,
                                [i, q, p],
                                back_prop=False,
                                parallel_iterations=1)
        return q, p

    @add_name_scope
    def _adapt_step_size(self, acceptance_rate, if_initialize_step_size):
        new_step_size = self.step_size_tuner.tune(
            tf.reduce_mean(acceptance_rate),
            tf.cast(if_initialize_step_size, tf.float32))
        update_step_size = tf.assign(self.step_size, new_step_size)
        return tf.stop_gradient(update_step_size)

    def sample(self, meta_bn, observed, latent):
        """
        Return the sampling `Operation` that runs a HMC iteration and
        the statistics collected during it.

        :param log_joint: A function that accepts a dictionary argument of
            ``(string, Tensor)`` pairs, which are mappings from all
            `StochasticTensor` names in the model to their observed values. The
            function should return a Tensor, representing the log joint
            likelihood of the model.
        :param observed: A dictionary of ``(string, Tensor)`` pairs. Mapping
            from names of observed `StochasticTensor` s to their values
        :param latent: A dictionary of ``(string, Variable)`` pairs.
            Mapping from names of latent `StochasticTensor` s to corresponding
            tensorflow Variables for storing their initial values and samples.

        :return: A Tensorflow `Operation` that runs a HMC iteration.
        :return: A :class:`HMCInfo` instance that collects sampling statistics
            during an iteration.
        """

        if callable(meta_bn):
            # TODO: raise warning
            self._meta_bn = None
            self._log_joint = meta_bn
        else:
            self._meta_bn = meta_bn
            self._log_joint = lambda obs: meta_bn.observe(**obs).log_joint()

        self._latent = latent
        self._observed = observed

        new_t = self.t.assign_add(1.0)
        latent_k, latent_v = [list(i) for i in zip(*six.iteritems(latent))]
        for i, v in enumerate(latent_v):
            if not isinstance(v, tf.Variable):
                raise TypeError("latent['{}'] is not a tensorflow Variable."
                                .format(latent_k[i]))
        self.q = copy(latent_v)

        def get_log_posterior(var_list):
            joint_obs = merge_dicts(dict(zip(latent_k, var_list)), observed)
            return self._log_joint(joint_obs)

        def get_gradient(var_list):
            log_p = get_log_posterior(var_list)
            return tf.gradients(log_p, var_list)

        self.static_shapes = [q.get_shape() for q in self.q]
        self.dynamic_shapes = [tf.shape(q) for q in self.q]
        self.static_chain_shape = get_log_posterior(self.q).get_shape()

        if not self.static_chain_shape:
            raise ValueError(
                "HMC requires that the static shape of the value returned "
                "by log joint function should be at least partially defined. "
                "(shape: {})".format(self.static_chain_shape))

        self.n_chain_dims = len(self.static_chain_shape)
        self.data_shapes = [
            tf.TensorShape([1] * self.n_chain_dims).concatenate(
                q.get_shape()[self.n_chain_dims:]) for q in self.q]
        self.data_axes = [list(range(self.n_chain_dims, len(data_shape)))
                          for data_shape in self.data_shapes]

        # Adapt mass
        if self.adapt_mass is not None:
            mass = [tf.stop_gradient(t) for t in
                    self._adapt_mass(new_t, self.n_chain_dims)]
        else:
            mass = [tf.ones(shape) for shape in self.data_shapes]

        p = random_momentum(self.dynamic_shapes, mass)
        current_p = copy(p)
        current_q = copy(self.q)

        # Initialize step size
        if self.adapt_step_size is None:
            new_step_size = self.step_size
        else:
            if_initialize_step_size = tf.logical_or(tf.equal(new_t, 1),
                tf.equal(tf.cast(new_t, tf.int32), self.mass_collect_iters))
            def iss():
                return self._init_step_size(current_q, current_p, mass,
                                            get_gradient, get_log_posterior)
            new_step_size = tf.stop_gradient(
                tf.cond(if_initialize_step_size, iss, lambda: self.step_size))

        # Leapfrog
        current_q, current_p = self._leapfrog(
            current_q, current_p, new_step_size, get_gradient, mass)

        # MH-Test
        with tf.name_scope("MH-test"):
            old_hamiltonian, new_hamiltonian, old_log_prob, new_log_prob, \
                acceptance_rate = get_acceptance_rate(
                    self.q, p, current_q, current_p,
                    get_log_posterior, mass, self.data_axes)

            u01 = tf.random_uniform(shape=tf.shape(acceptance_rate))
            if_accept = tf.less(u01, acceptance_rate)

            new_q = []
            for nq, oq, da in zip(current_q, self.q, self.data_axes):
                expanded_if_accept = if_accept
                for i in range(len(da)):
                    expanded_if_accept = tf.expand_dims(expanded_if_accept, -1)
                expanded_if_accept = tf.logical_and(
                    expanded_if_accept, tf.ones_like(nq, dtype=tf.bool))
                new_q.append(tf.where(expanded_if_accept, nq, oq))

            update_q = [old.assign(new) for old, new in zip(latent_v, new_q)]
            new_log_prob = tf.where(if_accept, new_log_prob, old_log_prob)

        # Adapt step size
        if self.adapt_step_size is not None:
            update_step_size = self._adapt_step_size(acceptance_rate,
                                                     if_initialize_step_size)
        else:
            update_step_size = self.step_size

        # Pack HMC statistics
        hmc_info = HMCInfo(
            samples=dict(zip(latent_k, new_q)),
            acceptance_rate=acceptance_rate,
            updated_step_size=update_step_size,
            init_momentum=dict(zip(latent_k, p)),
            orig_hamiltonian=old_hamiltonian,
            hamiltonian=new_hamiltonian,
            orig_log_prob=old_log_prob,
            log_prob=new_log_prob,
        )

        with tf.control_dependencies([update_step_size]):
            sample_op = tf.group(*update_q)

        return sample_op, hmc_info

    @property
    def bn(self):
        try:
            if self._meta_bn:
                return self._meta_bn.observe(
                    **merge_dicts(self._latent, self._observed))
            else:
                return None
        except AttributeError:
            return None
