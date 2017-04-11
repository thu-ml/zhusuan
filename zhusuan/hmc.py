#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from copy import copy

import six
from six.moves import zip, map, filter
import tensorflow as tf

from zhusuan.utils import add_name_scope, merge_dicts


__all__ = [
    "HMC",
]


def random_momentum(dshapes, mass):
    return [tf.random_normal(shape=shape) * tf.sqrt(m)
            for shape, m in zip(dshapes, mass)]


def velocity(momentum, mass):
    return map(lambda z: z[0] / z[1], zip(momentum, mass))
    # return map(lambda (x, y): x / y, zip(momentum, mass))


def hamiltonian(q, p, log_posterior, mass, data_axes):
    # (n_particles, n)
    potential = -log_posterior(q)
    # (n_particles, n)
    kinetic = 0.5 * tf.add_n(
        [tf.reduce_sum(tf.square(momentum) / m, axis)
         for momentum, m, axis in zip(p, mass, data_axes)])
    # (n_particles, n)
    return potential + kinetic, -potential
    # return tf.reshape(potential, tf.shape(kinetic)) + \
    #        tf.Print(kinetic, [q, p, potential, kinetic])


def leapfrog_integrator(q, p, step_size1, step_size2, grad, mass):
    # [(n_particles, n, n_z)]
    q = [x + step_size1 * y for x, y in zip(q, velocity(p, mass))]
    # p = p + epsilon / 2 * gradient q
    # [(n_particles, n, n_z)]
    grads = grad(q)
    # [(n_particles, n, n_z)]
    p = [x + step_size2 * y for x, y in zip(p, grads)]
    return q, p


def get_acceptance_rate(q, p, new_q, new_p, log_posterior, mass, data_axes):
    # (n_particles, n)
    old_hamiltonian, old_log_prob = hamiltonian(
        q, p, log_posterior, mass, data_axes)
    # (n_particles, n)
    new_hamiltonian, new_log_prob = hamiltonian(
        new_q, new_p, log_posterior, mass, data_axes)
    # (n_particles, n)
    return old_hamiltonian, new_hamiltonian, old_log_prob, new_log_prob, \
        tf.exp(tf.minimum(-new_hamiltonian + old_hamiltonian, 0.0))


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
            rate1 = tf.div(1.0, new_step + self.t0)
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
            self.chain_axes = range(self.num_chain_dims)

    @add_name_scope
    def update(self, x):
        # x: (chain_dims data_dims)
        new_t = tf.assign(self.t, self.t + 1)
        weight = (1 - self.decay) / (1 - tf.pow(self.decay, new_t))
        # incr: (chain_dims data_dims)
        incr = [weight * (q - mean) for q, mean in zip(x, self.mean)]
        # mean: (1,...,1 data_dims)
        update_mean = [mean.assign_add(
            tf.reduce_mean(i, list(self.chain_axes), keep_dims=True))
                       for mean, i in zip(self.mean, incr)]
        # var: (1,...,1 data_dims)
        new_var = [
            (1 - weight) * var +
            tf.reduce_mean(i * (q - mean), list(self.chain_axes),
                           keep_dims=True)
            for var, i, q, mean in zip(self.var, incr, x, update_mean)]

        update_var = [tf.assign(var, n_var)
                      for var, n_var in zip(self.var, new_var)]
        return update_var

    def get_precision(self, var_in):
        return [tf.div(self.one, var) for var in var_in]

    def get_updated_precision(self, x):
        # Should be called only once
        return self.get_precision(self.update(x))

    def precision(self):
        return self.get_precision(self.var)


class HMC:
    def __init__(self, step_size=1, n_leapfrogs=10,
                 adapt_step_size=None, target_acceptance_rate=0.8,
                 gamma=0.05, t0=100, kappa=0.75,
                 adapt_mass=None, mass_collect_iters=10, mass_decay=0.99):
        # TODO: Maintain the variables somewhere else to let the sample be
        # called multiple times
        self.step_size = tf.Variable(step_size, name="step_size",
                                     trainable=False)
        self.n_leapfrogs = tf.convert_to_tensor(n_leapfrogs,
                                                name="n_leapfrogs")
        self.target_acceptance_rate = tf.convert_to_tensor(
            target_acceptance_rate, name="target_acceptance_rate")
        self.t = tf.Variable(0.0, dtype=tf.float32, name="t",
                             trainable=False)
        self.adapt_step_size = adapt_step_size
        if adapt_step_size is not None:
            # TODO make sure adapt_step_size is a placeholder
            self.step_size_tuner = StepsizeTuner(
                step_size, adapt_step_size, gamma, t0, kappa,
                target_acceptance_rate)
        if adapt_mass is not None:
            self.adapt_mass = tf.convert_to_tensor(
                adapt_mass, dtype=tf.bool, name="adapt_mass")
        else:
            mass_collect_iters = 0
            self.adapt_mass = None
        self.mass_collect_iters = mass_collect_iters
        self.mass_decay = mass_decay

    @add_name_scope
    def _adapt_mass(self, t, num_chain_dims):
        ewmv = ExponentialWeightedMovingVariance(
            self.mass_decay, self.data_shape, num_chain_dims)
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
            current_mass = tf.cond(tf.less(t, self.mass_collect_iters),
                                   lambda: [tf.ones(shape) for shape in
                                            self.data_shape],
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
            # return [tf.Print(new_step_size,
            #                   [new_step_size, acceptance_rate], "Tuning"),
            #          acceptance_rate, cond]

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

            # [(n_particles, n, n_z)], [(n_particles, n, n_z)]
            q, p = leapfrog_integrator(q, p, step_size1, step_size2,
                                       lambda q: get_gradient(q), mass)
            return [i + 1, q, p]

        i = tf.constant(0)
        # [(n_particles, n, n_z)], [(n_particles, n, n_z)]
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

    # Shape = [ChainShape DataShape]
    # Data shape should not change
    def sample(self, log_joint, observed, latent):
        """
        sample(log_joint, observed, latent)
        """
        new_t = self.t.assign_add(1.0)
        latent_k, latent_v = [list(i) for i in zip(*six.iteritems(latent))]

        self.q = copy(latent_v)
        # self.q = [tf.Print(self.q[0], [self.q[0]], "q")]

        def get_log_posterior(var_list):
            # (chain_axis)
            joint_obs = merge_dicts(dict(zip(latent_k, var_list)), observed)
            log_p = log_joint(joint_obs)
            return log_p

        def get_gradient(var_list):
            log_p = get_log_posterior(var_list)
            # (chain_axis data_axis)
            latent_grads = tf.gradients(log_p, var_list)
            # print('LG = {}'.format(latent_grads))
            return latent_grads

        self.sshapes = [q.get_shape() for q in self.q]
        self.dshapes = [tf.shape(q) for q in self.q]
        self.s_chain_shape = get_log_posterior(self.q).get_shape()
        self.num_chain_dims = len(self.s_chain_shape)
        # [1, .., 1, data_dims]
        self.data_shape = [
            tf.reduce_sum(q, axis=list(range(self.num_chain_dims)),
                          keep_dims=True).get_shape() for q in self.q]
        self.data_axes = [list(range(self.num_chain_dims, len(shape)))
                          for shape in self.data_shape]

        # print('Static shape = {}'.format(self.sshapes))
        # print('Data shape = {}'.format(self.data_shape))
        # print('Num chain dims = {}, data axes = {}'.
        #       format(self.num_chain_dims, self.data_axes))

        if self.adapt_mass is not None:
            mass = [tf.stop_gradient(t) for t in
                    self._adapt_mass(new_t, self.num_chain_dims)]
        else:
            mass = [tf.ones(shape) for shape in self.data_shape]

        # print('Mass shape = {}'.format([m.get_shape() for m in mass]))

        # print('Current mass shape={}'.format(current_mass[0].get_shape()))
        # print('Expanded mass shape={}'.format(expanded_mass[0].get_shape()))

        p = random_momentum(self.dshapes, mass)
        # print('P shape = {}'.format([pp.get_shape() for pp in p]))
        # p = [tf.Print(p[0], [p[0]], "p")]

        current_p = copy(p)
        current_q = copy(self.q)
        # print('p shape = {}'.format(current_p[0].get_shape()))
        # print('q shape = {}'.format(current_q[0].get_shape()))

        # Initialize step size
        if_initialize_step_size = tf.logical_or(
            tf.equal(new_t, 1), tf.equal(new_t, self.mass_collect_iters))

        def iss():
            return self._init_step_size(current_q, current_p, mass,
                                        get_gradient, get_log_posterior)
        new_step_size = tf.stop_gradient(
            tf.cond(if_initialize_step_size, iss, lambda: self.step_size)
        )
        # new_step_size = tf.Print(new_step_size, [new_step_size])

        # Leapfrog
        current_q, current_p = self._leapfrog(
            current_q, current_p, new_step_size, get_gradient, mass)

        # for q in current_q:
        #     print(q.get_shape())
        # for pp in current_p:
        #     print(pp.get_shape())
        # for q in self.q:
        #     print(q.get_shape())
        # for pp in p:
        #     print(pp.get_shape())
        # current_q = [tf.Print(current_q[0], [current_q[0]], "cq")]
        # current_p = [tf.Print(current_p[0], [current_p[0]], "cp")]

        # (n_particles, n)
        with tf.name_scope("MH-test"):
            old_hamiltonian, new_hamiltonian, old_log_prob, new_log_prob, \
                acceptance_rate = get_acceptance_rate(
                    self.q, p, current_q, current_p,
                    get_log_posterior, mass, self.data_axes)

            # (n_particles, n)
            u01 = tf.random_uniform(shape=tf.shape(acceptance_rate))
            if_accept = tf.less(u01, acceptance_rate)

            # print('Acceptance rate shape = {}'.format(
            #     acceptance_rate.get_shape()))

            new_q = []
            for nq, oq, da in zip(current_q, self.q, self.data_axes):
                expanded_if_accept = if_accept
                for i in range(len(da)):
                    expanded_if_accept = tf.expand_dims(expanded_if_accept, -1)
                expanded_if_accept = tf.logical_and(
                    expanded_if_accept, tf.ones_like(nq, dtype=tf.bool)
                )
                # print('Expanded if accept shape = {}'
                #     .format(expanded_if_accept.get_shape()))
                new_q.append(tf.where(expanded_if_accept, nq, oq))

            update_q = [old.assign(new) for old, new in zip(latent_v, new_q)]

            new_log_prob = tf.where(if_accept, new_log_prob, old_log_prob)

        if self.adapt_step_size is not None:
            update_step_size = self._adapt_step_size(acceptance_rate,
                                                     if_initialize_step_size)
        else:
            update_step_size = self.step_size

        with tf.control_dependencies([update_step_size]):
            return update_q, p, tf.squeeze(old_hamiltonian), \
                tf.squeeze(new_hamiltonian), old_log_prob, new_log_prob, \
                tf.squeeze(acceptance_rate), update_step_size
