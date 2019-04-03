#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import six
from six.moves import zip
from copy import copy
from collections import namedtuple
import tensorflow as tf

from zhusuan.utils import merge_dicts


__all__ = [
    "SGMCMC",
    "SGLD",
    "PSGLD",
    "SGHMC",
    "SGNHT",
]


class SGMCMC(object):
    """
    Base class for stochastic gradient MCMC algorithms.
    """
    def __init__(self):
        self.t = tf.Variable(0, name="t", trainable=False, dtype=tf.int32)

    def make_get_gradient(self, meta_model, observed, latent):
        # TODO: remove direct; use tf.greater
        if callable(meta_model):
            # TODO: raise warning
            self._meta_model = None
            self._log_joint = meta_model
        else:
            self._meta_model = meta_model
            self._log_joint = lambda obs: meta_model.observe(**obs).log_joint()

        self._observed = observed
        self._latent = latent

        latent_k, latent_v = [list(i) for i in zip(*six.iteritems(latent))]
        for i, v in enumerate(latent_v):
            if not isinstance(v, tf.Variable):
                raise TypeError("latent['{}'] is not a tensorflow Variable."
                                .format(latent_k[i]))

        def _get_log_posterior(var_list, observed):
            joint_obs = merge_dicts(dict(zip(latent_k, var_list)), observed)
            return self._log_joint(joint_obs)

        def _get_gradient(var_list, observed):
            return tf.gradients(_get_log_posterior(var_list, observed), var_list)

        self._default_get_gradient = lambda var_list: _get_gradient(var_list, observed)
        self._latent_k = latent_k
        self._var_list = latent_v
        return self._default_get_gradient, _get_gradient, latent_v

    def sample(self, grad_func=None):
        qs = copy(self._var_list)
        self._define_variables(qs)
        if grad_func is None:
            grad_func = self._default_get_gradient
        update_ops, new_qs, infos = self._update(qs, grad_func)

        with tf.control_dependencies([self.t.assign_add(1)]):
            sample_op = tf.group(*update_ops)
        new_samples = dict(zip(self._latent_k, new_qs))
        sample_info = dict(zip(self._latent_k, infos))
        return sample_op, new_samples, sample_info

    def _update(self, qs, grad_func):
        return NotImplementedError()
    
    def _define_variables(self, qs):
        return NotImplementedError()

    @property
    def bn(self):
        if hasattr(self, "_meta_model"):
            if self._meta_model:
                if not hasattr(self, "_bn"):
                    self._bn = self._meta_model.observe(
                        **merge_dicts(self._latent, self._observed))
                return self._bn
            else:
                return None
        else:
            return None


class SGLD(SGMCMC):
    """
    Stochastic Gradient Langevin Dynamics
    """
    def __init__(self, learning_rate=0.1, add_noise=True):
        self.lr = tf.convert_to_tensor(learning_rate, tf.float32,
                                       name="learning_rate")
        if type(add_noise) == bool:
            add_noise = tf.constant(add_noise)
        self.add_noise = add_noise
        super(SGLD, self).__init__()

    def _define_variables(self, qs):
        pass

    def _update(self, qs, grad_func):
        return zip(*[self._update_single(q, grad) for q, grad in zip(qs, grad_func(qs))])

    def _update_single(self, q, grad):
        new_q = q + 0.5 * self.lr * grad
        new_q_with_noise = new_q + tf.random_normal(
            tf.shape(q), stddev=tf.sqrt(self.lr))
        new_q = tf.cond(self.add_noise, lambda: new_q_with_noise, lambda: new_q)
        update_q = q.assign(new_q)
        return update_q, new_q, tf.constant("No info.")


class PSGLD(SGLD):
    """
    Preconditioned Stochastic Gradient Langevin Dynamics
    """

    class RMSPreconditioner:
        HParams = namedtuple('RMSHParams', 'decay epsilon')
        default_hps = HParams(decay=0.9, epsilon=1e-3)
        @staticmethod
        def _define_variables(qs):
            return [tf.Variable(tf.zeros_like(q)) for q in qs]
        @staticmethod
        def _get_preconditioner(hps, q, grad, aux):
            aux = tf.assign(aux, hps.decay * aux + (1-hps.decay) * grad**2)
            return 1 / (hps.epsilon + tf.sqrt(aux))

    def __init__(self, learning_rate=0.1, add_noise=True,
                 preconditioner='rms', preconditioner_hparams=None):
        self.preconditioner = {
            'rms': PSGLD.RMSPreconditioner
        }[preconditioner]
        if preconditioner_hparams is None:
            preconditioner_hparams = self.preconditioner.default_hps
        self.preconditioner_hparams = preconditioner_hparams
        super(PSGLD, self).__init__(learning_rate, add_noise)

    def _define_variables(self, qs):
        self.vs = self.preconditioner._define_variables(qs)

    def _update(self, qs, grad_func):
        return zip(*[self._update_single(q, grad, aux)
                     for q, grad, aux in zip(qs, grad_func(qs), self.vs)])

    def _update_single(self, q, grad, aux):
        g = self.preconditioner._get_preconditioner(
            self.preconditioner_hparams, q, grad, aux)
        new_q = q + 0.5 * self.lr * g * grad
        new_q_with_noise = new_q + tf.random_normal(
            tf.shape(q), stddev=tf.sqrt(self.lr * g))
        new_q = tf.cond(self.add_noise, lambda: new_q_with_noise, lambda: new_q)
        update_q = q.assign(new_q)
        return update_q, new_q, tf.constant("No info.")


class SGHMC(SGMCMC):
    """
    Stochastic Gradient Hamiltonian Monte Carlo
    """
    def __init__(self, learning_rate=0.1, friction=0.25, variance_estimate=0.,
                 n_iter_resample_v=20, second_order=True):
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
        self.second_order = second_order
        super(SGHMC, self).__init__()

    def _define_variables(self, qs):
        self.vs = [tf.Variable(tf.random_normal(tf.shape(q), stddev=tf.sqrt(self.lr))) for q in qs]

    def _update(self, qs, grad_func):
        def resample_momentum(v):
            return tf.random_normal(tf.shape(v), stddev=tf.sqrt(self.lr))

        old_vs = [tf.cond(tf.equal(self.n_iter_resample_v, 0), lambda: v, lambda: tf.cond(
            tf.equal(tf.mod(self.t, self.n_iter_resample_v), 0),
            lambda: resample_momentum(v), lambda: v)) for v in self.vs]
        gaussian_terms = [tf.random_normal(
            tf.shape(old_v), stddev=tf.sqrt(2*(self.alpha-self.beta)*self.lr)) for old_v in old_vs]
        if not self.second_order:
            new_vs = [(1 - self.alpha) * old_v + self.lr * grad + gaussian_term
                for (old_v, grad, gaussian_term) in zip(old_vs, grad_func(qs), gaussian_terms)]
            new_qs = [q + new_v for (q, new_v) in zip(qs, new_vs)]
        else:
            decay_half = tf.exp(-0.5*self.alpha)
            q1s = [q + 0.5 * old_v for (q, old_v) in zip(qs, old_vs)]
            new_vs = [decay_half * (decay_half * old_v + self.lr * grad + gaussian_term)
                for (old_v, grad, gaussian_term) in zip(old_vs, grad_func(q1s), gaussian_terms)]
            new_qs = [q1 + 0.5 * new_v for (q1, new_v) in zip(q1s, new_vs)]

        mean_ks = [tf.reduce_mean(new_v**2) for new_v in new_vs]
        infos = [{"mean_k": mean_k} for mean_k in mean_ks]

        with tf.control_dependencies(new_vs + new_qs):
            update_qs = [q.assign(new_q) for (q, new_q) in zip(qs, new_qs)]
            update_vs = [v.assign(new_v) for (v, new_v) in zip(self.vs, new_vs)]

        update_ops = [tf.group(update_q, update_v)
            for (update_q, update_v) in zip(update_qs, update_vs)]

        return update_ops, new_qs, infos


class SGNHT(SGMCMC):
    """
    Stochastic Gradient Nosé-Hoover Thermostat
    """
    def __init__(self, learning_rate=0.1, variance_extra=0., tune_rate=1.,
                 n_iter_resample_v=None, second_order=True, use_vector_xi=True):
        self.lr = tf.convert_to_tensor(learning_rate, tf.float32,
                                        name="learning_rate")
        self.alpha = tf.convert_to_tensor(variance_extra, tf.float32,
                                          name="variance_extra")
        self.tune_rate = tf.convert_to_tensor(tune_rate, tf.float32,
                                          name="tune_rate")
        if n_iter_resample_v is None:
            n_iter_resample_v = 0
        self.n_iter_resample_v = tf.convert_to_tensor(n_iter_resample_v, tf.int32,
                                                      name="n_iter_resample_v")
        self.second_order = second_order
        self.use_vector_xi = use_vector_xi
        super(SGNHT, self).__init__()

    def _define_variables(self, qs):
        self.vs = [tf.Variable(tf.random_normal(tf.shape(q), stddev=tf.sqrt(self.lr))) for q in qs]
        if self.use_vector_xi:
            self.xis = [tf.Variable(self.alpha*tf.ones(tf.shape(q))) for q in qs]
        else:
            self.xis = [tf.Variable(self.alpha) for q in qs]

    def _update(self, qs, grad_func):
        def resample_momentum(v):
            return tf.random_normal(tf.shape(v), stddev=tf.sqrt(self.lr))
        
        def maybe_reduce_mean(tensor):
            if self.use_vector_xi:
                return tensor
            else:
                return tf.reduce_mean(tensor)

        old_vs = [tf.cond(tf.equal(self.n_iter_resample_v, 0), lambda: v, lambda: tf.cond(
            tf.equal(tf.mod(self.t, self.n_iter_resample_v), 0),
            lambda: resample_momentum(v), lambda: v)) for v in self.vs]
        gaussian_terms = [tf.random_normal(
            tf.shape(old_v), stddev=tf.sqrt(2*self.alpha*self.lr)) for old_v in old_vs]
        if not self.second_order:
            new_vs = [(1 - xi) * old_v + self.lr * grad + gaussian_term
                for (old_v, xi, grad, gaussian_term) in zip(old_vs, self.xis, grad_func(qs), gaussian_terms)]
            new_qs = [q + new_v for (q, new_v) in zip(qs, new_vs)]
            mean_ks = [maybe_reduce_mean(new_v**2) for new_v in new_vs]
            new_xis = [xi + self.tune_rate * (mean_k - self.lr) for (xi, mean_k) in zip(self.xis, mean_ks)]
        else:
            q1s = [q + 0.5 * old_v for (q, old_v) in zip(qs, old_vs)]
            mean_k1s = [maybe_reduce_mean(old_v**2) for old_v in old_vs]
            xi1s = [xi + 0.5 * self.tune_rate * (mean_k1 - self.lr) for (xi, mean_k1) in zip(self.xis, mean_k1s)]
            decay_halfs = [tf.exp(-0.5*xi1) for xi1 in xi1s]
            new_vs = [decay_half * (decay_half * old_v + self.lr * grad + gaussian_term)
                for (decay_half, old_v, grad, gaussian_term) in zip(decay_halfs, old_vs, grad_func(q1s), gaussian_terms)]
            new_qs = [q1 + 0.5 * new_v for (q1, new_v) in zip(q1s, new_vs)]
            mean_ks = [maybe_reduce_mean(new_v**2) for new_v in new_vs]
            new_xis = [xi1 + 0.5 * self.tune_rate * (mean_k - self.lr) for (xi1, mean_k) in zip(xi1s, mean_ks)]

        infos = [{"mean_k": tf.reduce_mean(mean_k), "xi": tf.reduce_mean(new_xi)} for (mean_k, new_xi) in zip(mean_ks, new_xis)]

        with tf.control_dependencies(new_vs + new_qs + new_xis):
            update_qs = [q.assign(new_q) for (q, new_q) in zip(qs, new_qs)]
            update_vs = [v.assign(new_v) for (v, new_v) in zip(self.vs, new_vs)]
            update_xis = [xi.assign(new_xi) for (xi, new_xi) in zip(self.xis, new_xis)]

        update_ops = [tf.group(update_q, update_v, update_xi)
            for (update_q, update_v, update_xi) in zip(update_qs, update_vs, update_xis)]

        return update_ops, new_qs, infos
