#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Stochastic variational inference for sparse Gaussian process (SVGP).

For the formulation you can refer to, e.g., Section 2.1 of the following paper:

Salimbeni and Deisenroth 2017, Doubly Stochastic Variational Inference for Deep
Gaussian Processes.

Results (mean and std.dev.) with 100 inducing points:

Dataset        RMSE           NLL        n_epochs     lr
--------  -------------  -------------  ----------    ----
Boston     2.90 (0.40)    2.52 (0.10)    2000         0.02
Protein    4.49 (0.03)    2.93 (0.01)    400          0.01
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import argparse

import numpy as np
from six.moves import range
import tensorflow as tf
import zhusuan as zs

from examples import conf
from examples.utils import dataset
from examples.gaussian_process.utils import gp_conditional, RBFKernel


parser = argparse.ArgumentParser()
parser.add_argument('-n_z', default=100, type=int)
parser.add_argument('-n_particles', default=20, type=int)
parser.add_argument('-n_particles_test', default=100, type=int)
parser.add_argument('-batch_size', default=5000, type=int)
parser.add_argument('-n_epoch', default=5000, type=int)
parser.add_argument('-dtype', default='float32', type=str,
                    choices=['float32', 'float64'])
parser.add_argument('-dataset', default='boston_housing', type=str,
                    choices=['boston_housing', 'protein_data'])
parser.add_argument('-lr', default=1e-2, type=float)


@zs.reuse('model')
def build_model(observed, hps, kernel, z_pos, x, n_particles, full_cov=False):
    """
    Build the SVGP model.
    Note that for inference, we only need the diagonal part of Cov[Y], as
    ELBO equals sum over individual observations.
    For visualization etc we may want a full covariance. Thus the argument
    `full_cov`.
    """
    with zs.BayesianNet(observed) as model:
        Kzz_chol = tf.cholesky(kernel(z_pos, z_pos))
        fz = zs.MultivariateNormalCholesky(
            'fz', tf.zeros([hps.n_z], dtype=hps.dtype), Kzz_chol,
            n_samples=n_particles)
        # f(X)|f(Z) follows GP(0, K) gp_conditional
        fx_given_fz = gp_conditional(
            z_pos, fz, x, full_cov, kernel, 'fx', Kzz_chol)
        # Y|f(X) ~ N(f(X), noise_level * I)
        noise_level = tf.get_variable(
            'noise_level', shape=[], dtype=hps.dtype,
            initializer=tf.constant_initializer(0.05))
        noise_level = tf.nn.softplus(noise_level)
        y = zs.Normal(
            'y', mean=fx_given_fz, std=noise_level, group_ndims=1)
    return model, y


def build_variational(hps, kernel, z_pos, x, n_particles):
    with zs.BayesianNet() as variational:
        z_mean = tf.get_variable(
            'z/mean', [hps.n_z], hps.dtype, tf.zeros_initializer())
        z_cov_raw = tf.get_variable(
            'z/cov_raw', initializer=tf.eye(hps.n_z, dtype=hps.dtype))
        z_cov_tril = tf.matrix_set_diag(
            tf.matrix_band_part(z_cov_raw, -1, 0),
            tf.nn.softplus(tf.matrix_diag_part(z_cov_raw)))
        fz = zs.MultivariateNormalCholesky(
            'fz', z_mean, z_cov_tril, n_samples=n_particles)
        fx_given_fz = gp_conditional(
            z_pos, fz, x, False, kernel, 'fx')
    return variational


def main():
    # tf.set_random_seed(1237)
    # np.random.seed(1234)
    hps = parser.parse_args()

    # Load data
    data_path = os.path.join(conf.data_dir, hps.dataset + '.data')
    data_func = getattr(dataset, 'load_uci_' + hps.dataset)
    x_train, y_train, x_valid, y_valid, x_test, y_test = data_func(data_path)
    x_train = np.vstack([x_train, x_valid])
    y_train = np.hstack([y_train, y_valid])
    n_train, n_covariates = x_train.shape
    hps.dtype = getattr(tf, hps.dtype)

    # Standardize data
    x_train, x_test, _, _ = dataset.standardize(x_train, x_test)
    y_train, y_test, mean_y_train, std_y_train = dataset.standardize(
        y_train, y_test)

    # Build model
    kernel = RBFKernel(n_covariates)
    x_ph = tf.placeholder(hps.dtype, [None, n_covariates], 'x')
    y_ph = tf.placeholder(hps.dtype, [None], 'y')
    z_pos = tf.get_variable(
        'z/pos', [hps.n_z, n_covariates], hps.dtype,
        initializer=tf.random_uniform_initializer(-1, 1))
    n_particles_ph = n_particles_ph = tf.placeholder(
        tf.int32, [], 'n_particles')
    batch_size = tf.cast(tf.shape(x_ph)[0], hps.dtype)

    # Training ops
    # ELBO = E_q log (p(y|fx)p(fx|fz)p(fz) / p(fx|fz)q(fz))
    # So we remove p(fx|fz) in both log_joint and latent
    def log_joint(observed):
        model, _ = build_model(
            observed, hps, kernel, z_pos, x_ph, n_particles_ph)
        prior, log_py_given_fx = model.local_log_prob(['fz', 'y'])
        return prior + log_py_given_fx / batch_size * n_train

    variational = build_variational(hps, kernel, z_pos, x_ph, n_particles_ph)
    [var_fz, var_fx] = variational.query(
        ['fz', 'fx'], outputs=True, local_log_prob=True)
    var_fx = (var_fx[0], tf.zeros_like(var_fx[1]))
    lower_bound = zs.variational.elbo(
        log_joint,
        observed={'y': y_ph},
        latent={'fz': var_fz, 'fx': var_fx},
        axis=0)
    cost = lower_bound.sgvb()
    optimizer = tf.train.AdamOptimizer(learning_rate=hps.lr)
    infer_op = optimizer.minimize(cost)

    # Prediction ops
    observed = {'fx': var_fx[0], 'y': y_ph}
    model, y_inferred = build_model(
        observed, hps, kernel, z_pos, x_ph, n_particles_ph)
    log_likelihood = model.local_log_prob('y')
    std_y_train = tf.cast(std_y_train, hps.dtype)
    log_likelihood = zs.log_mean_exp(log_likelihood, 0) / batch_size - \
        tf.log(std_y_train)
    y_pred_mean = tf.reduce_mean(y_inferred.distribution.mean, axis=0)
    pred_mse = tf.reduce_mean((y_pred_mean - y_ph) ** 2) * std_y_train ** 2

    def infer_step(sess, x_batch, y_batch):
        fd = {
            x_ph: x_batch,
            y_ph: y_batch,
            n_particles_ph: hps.n_particles
        }
        return sess.run([infer_op, cost], fd)[1]

    def predict_step(sess, x_batch, y_batch):
        fd = {
            x_ph: x_batch,
            y_ph: y_batch,
            n_particles_ph: hps.n_particles_test
        }
        return sess.run([log_likelihood, pred_mse], fd)

    iters = int(np.ceil(x_train.shape[0] / float(hps.batch_size)))
    test_freq = 100
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(1, hps.n_epoch + 1):
            lbs = []
            indices = np.arange(x_train.shape[0])
            np.random.shuffle(indices)
            x_train = x_train[indices]
            y_train = y_train[indices]
            for t in range(iters):
                lb = infer_step(
                    sess,
                    x_train[t * hps.batch_size: (t + 1) * hps.batch_size],
                    y_train[t * hps.batch_size: (t + 1) * hps.batch_size])
                lbs.append(lb)
            if 10 * epoch % test_freq == 0:
                print('Epoch {}: Lower bound = {}'.format(epoch, np.mean(lbs)))
            if epoch % test_freq == 0:
                test_lls = []
                test_mses = []
                for t in range(0, x_test.shape[0], hps.batch_size):
                    ll, mse = predict_step(
                        sess,
                        x_test[t: t + hps.batch_size],
                        y_test[t: t + hps.batch_size])
                    test_lls.append(ll)
                    test_mses.append(mse)
                print('>> TEST')
                print('>> Test log likelihood = {}, rmse = {}'.format(
                    np.mean(test_lls), np.sqrt(np.mean(test_mses))))


if __name__ == '__main__':
    main()
