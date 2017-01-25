#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

from copy import copy

import six
from six.moves import zip, map, filter
import tensorflow as tf
import numpy as np

from zhusuan.utils import add_name_scope


def random_momentum(dshapes, mass):
    return [tf.random_normal(shape=shape) for shape,m in zip(dshapes, mass)]


def velocity(momentum, mass):
    return map(lambda (x, y): x / y, zip(momentum, mass))


def hamiltonian(q, p, log_posterior, mass, data_axis):
    # (n_particles, n)
    potential = -log_posterior(q)
    # (n_particles, n)
    kinetic = 0.5 * tf.add_n([tf.reduce_sum(tf.square(momentum) / m, axis, keep_dims=True)
                          for momentum, m, axis in zip(p, mass, data_axis)])
    # (n_particles, n)
    return tf.reshape(potential, tf.shape(kinetic)) + kinetic, -potential
#return tf.reshape(potential, tf.shape(kinetic)) + tf.Print(kinetic, [q, p, potential, kinetic])


def leapfrog_integrator(q, p, step_size1, step_size2, grad, mass):
    # [(n_particles, n, n_z)]
    q = [x + step_size1 * y for x, y in zip(q, velocity(p, mass))]
    # p = p + epsilon / 2 * gradient q
    # [(n_particles, n, n_z)]
    grads = grad(q)
    # [(n_particles, n, n_z)]
    p = [x + step_size2 * y for x, y in zip(p, grads)]
    return q, p


def get_acceptance_rate(q, p, new_q, new_p, log_posterior, mass, data_axis):
    # (n_particles, n)
    old_hamiltonian, old_log_prob = hamiltonian(q, p, log_posterior, mass, data_axis)
    # (n_particles, n)
    new_hamiltonian, new_log_prob = hamiltonian(new_q, new_p, log_posterior, mass, data_axis)
    # (n_particles, n)
    return old_hamiltonian, new_hamiltonian, old_log_prob, new_log_prob, \
        tf.exp(tf.minimum(-new_hamiltonian + old_hamiltonian, 0.0))


class StepsizeTuner:
    def __init__(self, initial_stepsize, adapt_step_size, gamma, t0, kappa, delta):
        with tf.name_scope("StepsizeTuner"):
            self.adapt_step_size = adapt_step_size

            self.gamma = tf.convert_to_tensor(gamma, dtype=tf.float32, name="gamma")
            self.t0 = tf.convert_to_tensor(t0, dtype=tf.float32, name="t0")
            self.kappa = tf.convert_to_tensor(kappa, dtype=tf.float32, name="kappa")
            self.delta = tf.convert_to_tensor(delta, dtype=tf.float32, name="delta")

            self.step = tf.Variable(0.0, dtype=tf.float32,
                                    name="step", trainable=False)
            self.log_epsilon_bar = tf.Variable(0.0, dtype=tf.float32,
                                               name="log_epsilon_bar", trainable=False)
            self.h_bar = tf.Variable(0.0, dtype=tf.float32,
                                     name="h_bar", trainable=False)
            self.mu = tf.Variable(10 * initial_stepsize, dtype=tf.float32,
                                  name="mu", trainable=False)

    @add_name_scope
    def tune(self, acceptance_rate):
        def adapt_stepsize():
            new_step = tf.assign(self.step, self.step + 1)
            rate1 = tf.div(1.0, new_step + self.t0)
            new_h_bar = tf.assign(self.h_bar, (1 - rate1) * self.h_bar +
                                  rate1 * (self.delta - acceptance_rate))
            log_epsilon = self.mu - tf.sqrt(new_step) / \
                self.gamma * new_h_bar
            rate = tf.pow(new_step, -self.kappa)
            new_log_epsilon_bar = tf.assign(
                self.log_epsilon_bar,
                rate * log_epsilon + (1 - rate) * self.log_epsilon_bar)
            with tf.control_dependencies([new_log_epsilon_bar]):
                new_log_epsilon = tf.identity(log_epsilon)

            return tf.exp(new_log_epsilon)

        c = tf.cond(self.adapt_step_size,
                    adapt_stepsize,
                    lambda: tf.exp(self.log_epsilon_bar))

        return c


def initialize_step_size(old_step_size, q, p, mass, get_gradient, get_log_posterior, 
                         data_axis, target_acceptance_rate):
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
            get_log_posterior, mass, data_axis)
    
        acceptance_rate = tf.reduce_mean(acceptance_rate)
    
        # Change step size and stopping criteria
        new_step_size = tf.cond(
            tf.less(acceptance_rate,
                    target_acceptance_rate),
            lambda: step_size * (1.0 / factor),
            lambda: step_size * factor)
    
        cond = tf.logical_not(tf.logical_xor(
            tf.less(last_acceptance_rate, target_acceptance_rate),
            tf.less(acceptance_rate, target_acceptance_rate)))
        return [new_step_size, acceptance_rate, cond]
        #return [tf.Print(new_step_size,
        #                  [new_step_size, acceptance_rate]),
        #         acceptance_rate, cond]
    
    new_step_size, _, _ = tf.while_loop(
        loop_cond,
        loop_body,
        [old_step_size, tf.constant(1.0), tf.constant(True)]
    )
    return new_step_size


class HMC:
    def __init__(self, step_size=1, n_leapfrogs=10,
                 adapt_step_size=None, target_acceptance_rate=0.8,
                 gamma=0.05, t0=100, kappa=0.75):
        with tf.name_scope("HMC"):
            self.step_size = tf.Variable(step_size, name="step_size", trainable=False)
            self.n_leapfrogs = tf.convert_to_tensor(n_leapfrogs,
                                                    name="n_leapfrogs")
            self.target_acceptance_rate = tf.convert_to_tensor(
                target_acceptance_rate, name="target_acceptance_rate")

            self.adapt_step_size = adapt_step_size
            self.t = tf.Variable(0.0, dtype=tf.float32, name="t")
            if adapt_step_size is not None: # TODO make sure adapt_step_size is a placeholder
                self.step_size_tuner = StepsizeTuner(step_size, adapt_step_size, gamma,
                                                 t0, kappa, target_acceptance_rate)

    # Shape = [ChainShape DataShape]
    # Data shape should not change
    #@add_name_scope
    def sample(self, log_posterior, observed, latent, given=None, chain_axis=None):
        new_t = self.t.assign_add(1.0)
        latent_k, latent_v = [list(i) for i in zip(*six.iteritems(latent))]

        self.q = copy(latent_v)
#self.q = [tf.Print(self.q[0], [self.q[0]], "q")]

        self.sshapes = [q.get_shape() for q in self.q]
        self.dshapes = [tf.shape(q) for q in self.q]
        self.data_axis = [[i for i in range(len(shape)) if i!=chain_axis]
                          for shape in self.sshapes]
        self.data_shape = [[shape[a] for a in axis] for axis, shape in
                           zip(self.data_axis, self.sshapes)]

        # print('Static shape = {}'.format(self.sshapes))
        # print('Data axis = {}'.format(self.data_axis))
        # print('Data shape = {}'.format(self.data_shape))

        current_mass = [tf.ones(shape) for shape in self.data_shape]
        expanded_mass = current_mass if chain_axis == None else \
                    [tf.expand_dims(m, chain_axis) for m in current_mass]

        # print('Current mass shape = {}'.format(current_mass[0].get_shape()))
        # print('Expanded mass shape = {}'.format(expanded_mass[0].get_shape()))

        p = random_momentum(self.dshapes, expanded_mass)
#p = [tf.Print(p[0], [p[0]], "p")]

        def get_log_posterior(var_list):
            # (chain_axis)
            log_p = log_posterior(dict(zip(latent_k, var_list)),
                                  observed, given)
            return log_p

        def get_gradient(var_list):
            log_p = get_log_posterior(var_list)

            # (chain_axis data_axis)
            latent_grads = tf.gradients(log_p, var_list)
            print('LG = {}'.format(latent_grads))
            return latent_grads

        current_p = copy(p)
        current_q = copy(self.q)
        # print('p shape = {}'.format(current_p[0].get_shape()))
        # print('q shape = {}'.format(current_q[0].get_shape()))

        # Initialize step size
        iss = lambda: initialize_step_size(self.step_size,
            current_q, current_p, current_mass,
            get_gradient, get_log_posterior, 
            self.data_axis, self.target_acceptance_rate)
        new_step_size = tf.cond(tf.equal(new_t, 1), iss, lambda: self.step_size)
        #new_step_size = tf.Print(new_step_size, [new_step_size])

        # Leapfrog
        def loop_cond(i, current_q, current_p):
            return i < self.n_leapfrogs + 1

        def loop_body(i, current_q, current_p):
            step_size1 = tf.cond(i > 0,
                                 lambda: new_step_size,
                                 lambda: tf.constant(0.0, dtype=tf.float32))

            step_size2 = tf.cond(tf.logical_and(tf.less(i, self.n_leapfrogs),
                                                tf.less(0, i)),
                                 lambda: new_step_size,
                                 lambda: new_step_size / 2)

            # [(n_particles, n, n_z)], [(n_particles, n, n_z)]
            current_q, current_p = leapfrog_integrator(
                current_q,
                current_p,
                step_size1,
                step_size2,
                lambda q: get_gradient(q),
                expanded_mass
            )
            return [i + 1, current_q, current_p]

        i = tf.constant(0)
        # [(n_particles, n, n_z)], [(n_particles, n, n_z)]
        _, current_q, current_p = tf.while_loop(loop_cond,
                                                loop_body,
                                                [i, current_q, current_p],
                                                back_prop=False,
                                                parallel_iterations=1)

        # for q in current_q:
        #     print(q.get_shape())
        # for pp in current_p:
        #     print(pp.get_shape())
        # for q in self.q:
        #     print(q.get_shape())
        # for pp in p:
        #     print(pp.get_shape())
#current_q = [tf.Print(current_q[0], [current_q[0]], "cq")]
#current_p = [tf.Print(current_p[0], [current_p[0]], "cp")]

        # (n_particles, n)
        old_hamiltonian, new_hamiltonian, old_log_prob, new_log_prob, acceptance_rate = \
            get_acceptance_rate(self.q, p, current_q, current_p,
                                get_log_posterior, current_mass, self.data_axis)

        # (n_particles, n)
        u01 = tf.random_uniform(shape=tf.shape(acceptance_rate))
        if_accept = tf.to_float(u01 < acceptance_rate)

        def myselect(condition, x, y):
            return condition * x + (1 - condition) * y

        new_q = [myselect(if_accept, x, y) for x, y in zip(current_q, self.q)]
        update_q = [old.assign(new) for old, new in zip(latent_v, new_q)]

        if self.adapt_step_size is not None:
            new_step_size = self.step_size_tuner.tune(tf.reduce_mean(acceptance_rate))
            update_step_size = tf.assign(self.step_size, new_step_size)
        else:
            update_step_size = self.step_size

        with tf.control_dependencies([update_step_size]):
            return update_q, p, tf.squeeze(old_hamiltonian), tf.squeeze(new_hamiltonian), \
               old_log_prob, new_log_prob, tf.squeeze(acceptance_rate), update_step_size
