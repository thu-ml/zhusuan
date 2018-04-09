#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from copy import copy

import six
from six.moves import zip
import tensorflow as tf

from zhusuan.utils import merge_dicts


__all__ = [
    "SGMCMC",
    "SGLD",
    "SGHMC",
    "SGNHT",
]


class SGMCMC:
    """
    Base class for stochastic gradient MCMC algorithms.
    """
    def __init__(self):
        self.t = tf.Variable(-1, name="t", trainable=False, dtype=tf.int32)

    def sample(self, log_joint, observed, latent):
        latent_k, latent_v = [list(i) for i in zip(*six.iteritems(latent))]
        for i, v in enumerate(latent_v):
            if not isinstance(v, tf.Variable):
                raise TypeError("latent['{}'] is not a tensorflow Variable."
                                .format(latent_k[i]))
        qs = copy(latent_v)
        self._define_variables(qs)

        def _log_joint(var_list):
            joint_obs = merge_dicts(dict(zip(latent_k, var_list)), observed)
            return log_joint(joint_obs)

        def _gradient(var_list):
            return tf.gradients(_log_joint(var_list), var_list)

        update_ops, new_qs = zip(*self._update(qs, _gradient))

        sample_op = tf.group(*update_ops)
        new_samples = dict(zip(latent_k, new_qs))
        return sample_op, new_samples

    def _update(self, qs, grad_func):
        return NotImplementedError()
    
    def _define_variables(self, qs):
        return NotImplementedError()


class SGLD(SGMCMC):
    """
    Stochastic Gradient Langevin Dynamics
    """
    def __init__(self, learning_rate=0.1):
        self.lr = tf.convert_to_tensor(learning_rate, tf.float32,
                                              name="learning_rate")
        super(SGLD, self).__init__()

    def _define_variables(self, qs):
        pass

    def _update(self, qs, grad_func):
        return [self._update_single(q, grad) for q, grad in zip(qs, grad_func(qs))]

    def _update_single(self, q, grad):
        new_q = q + 0.5 * self.lr * grad + \
                    tf.random_normal(tf.shape(q), stddev=tf.sqrt(self.lr))
        update_q = q.assign(new_q)
        return update_q, new_q


class SGHMC(SGMCMC):
    """
    Stochastic Gradient Hamiltonian Monte Carlo
    """
    def __init__(self, learning_rate=0.1, friction=0.25, variance_estimate=0.,
                 n_iter_resample_v=20):
        self.lr = tf.convert_to_tensor(learning_rate, tf.float32,
                                        name="learning_rate")
        self.alpha = tf.convert_to_tensor(friction, tf.float32,
                                          name="alpha")
        self.beta = tf.convert_to_tensor(variance_estimate, tf.float32,
                                         name="beta")
        if n_iter_resample_v is None:
            n_iter_resample_v = 0
        self.n_iter_resample_v = tf.convert_to_tensor(n_iter_resample_v, tf.int32,
                                                      name="n_iter_resample_v")
        super(SGHMC, self).__init__()

    def _define_variables(self, qs):
        self.vs = [tf.Variable(tf.zeros_like(q)) for q in qs]

    def _update(self, qs, grad_func):
        self.new_t = self.t.assign_add(1)
        return [self._update_single(q, v, grad)
            for q, v, grad in zip(qs, self.vs, grad_func(qs))]

    def _update_single(self, q, v, grad):
        def resample_momentum():
            return tf.random_normal(tf.shape(v), stddev=tf.sqrt(self.lr))

        old_v = tf.cond(tf.equal(tf.mod(self.new_t, self.n_iter_resample_v), 0),
            resample_momentum, lambda: v)

        new_v = (1 - self.alpha) * old_v + self.lr * grad + tf.random_normal(
            tf.shape(old_v), stddev=tf.sqrt(2*(self.alpha-self.beta)*self.lr))
        new_q = q + new_v

        with tf.control_dependencies([new_q, new_v]):
            update_q = q.assign(new_q)
            update_v = v.assign(new_v)
        update_op = tf.group(update_q, update_v)

        return update_op, new_q


class SGNHT(SGMCMC):
    """
    Stochastic Gradient Nosé-Hoover Thermostat
    """
    def __init__(self, learning_rate=0.1, variance_extra=0., tune_rate=1.):
        self.lr = tf.convert_to_tensor(learning_rate, tf.float32,
                                        name="learning_rate")
        self.alpha = tf.convert_to_tensor(variance_extra, tf.float32,
                                          name="variance_extra")
        self.tune_rate = tf.convert_to_tensor(tune_rate, tf.float32,
                                          name="tune_rate")
        super(SGNHT, self).__init__()

    def _define_variables(self, qs):
        self.vs = [tf.Variable(tf.zeros_like(q)) for q in qs]
        self.xis = [tf.Variable(self.alpha) for q in qs]

    def _update(self, qs, grad_func):
        self.new_t = self.t.assign_add(1)
        return [self._update_single(q, v, xi, grad)
            for q, v, xi, grad in zip(qs, self.vs, self.xis, grad_func(qs))]

    def _update_single(self, q, v, xi, grad):
        def sample_momentum():
            return tf.random_normal(tf.shape(v), stddev=tf.sqrt(self.lr))

        old_v = tf.cond(tf.equal(self.new_t, 0), sample_momentum, lambda: v)

        new_v = (1 - xi) * old_v + self.lr * grad + tf.random_normal(
            tf.shape(old_v), stddev=tf.sqrt(2*self.alpha*self.lr))
        new_q = q + new_v
        mean_k = tf.reduce_mean(new_v**2)
        # mean_k = tf.Print(mean_k, [mean_k])
        new_xi = xi + self.tune_rate * (mean_k - self.lr)
        # new_xi = tf.Print(new_xi, [new_xi])

        with tf.control_dependencies([new_q, new_v, new_xi]):
            update_q = q.assign(new_q)
            update_v = v.assign(new_v)
            update_xi = xi.assign(new_xi)
        update_op = tf.group(update_q, update_v, update_xi)

        return update_op, new_q

# TODO: 1. enable matrix parameter 2. enable mass setting