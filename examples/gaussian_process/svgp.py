#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Sparse variational Gaussian process with normal approximation for posterior.

For the formulation you can refer to, e.g., Section 2.1 of the following paper:

Salimbeni and Deisenroth 2017, Doubly Stochastic Variational Inference for Deep
Gaussian Processes.

On the Boston dataset you should get ~2.6 RMSE and ~2.3 negative log likelihood
after a while with default args;
On the protein dataset you should get ~4.5 RMSE and ~2.8 NLL after 100 epochs
with `n_batch=1000 N_particles=10 dtype=float32`.
'''

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from six.moves import range, zip

import zhusuan as zs
import tensorflow as tf
import numpy as np
from examples import conf
from examples.utils import dataset
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-n_z', default=100, type=int)
parser.add_argument('-n_covariates', default=1, type=int)
parser.add_argument('-n_particles', default=20, type=int)
parser.add_argument('-n_particles_test', default=100, type=int)
parser.add_argument('-n_batch', default=100, type=int)
parser.add_argument('-n_epoch', default=5000, type=int)
parser.add_argument('-dtype', default='float32', type=str,
                    choices=['float32', 'float64'])
parser.add_argument('-dataset', default='boston_housing', type=str,
                    choices=['boston_housing', 'protein_data'])


def rbf(x, y, inv_scale):
    x = tf.expand_dims(x, 1)
    y = tf.expand_dims(y, 0)
    inv_scale = tf.reshape(inv_scale, [1, 1, -1])
    return tf.exp(- tf.reduce_sum(tf.square(x - y) * inv_scale, axis=-1) / 2)


@zs.reuse('model')
def build_model(observed, x_ph, hps, n_particles, full_cov=False):
    '''
    Build the SVGP model.
    Note that for inference, we only need the diagonal part of Cov[Y], as ELBO
    equals sum over individual observations.
    For visualization etc we may want a full covariance. Thus the argument
    `full_cov`.
    '''
    with zs.BayesianNet(observed) as model:
        z_pos = tf.get_variable(
            'z/pos', [hps.n_z, hps.n_covariates], hps.dtype,
            initializer=tf.random_uniform_initializer(-1, 1))
        k_inv_scale = tf.get_variable(
            'k_inv_scale', [hps.n_covariates], hps.dtype,
            tf.zeros_initializer())
        k_inv_scale = tf.exp(k_inv_scale)

        Kzz = rbf(z_pos, z_pos, k_inv_scale)
        Kxz = rbf(x_ph, z_pos, k_inv_scale)
        Kzz_chol = tf.cholesky(Kzz)
        fZ = zs.MultivariateNormalCholesky(
            'fZ', tf.zeros([hps.n_z], dtype=hps.dtype), Kzz_chol,
            n_samples=n_particles)

        # Model Y|Z.
        # Mean[Y|Z] = Kxz @ inv(Kzz) @ fZ
        # Cov[Y|Z] = Kxx - Kxz @ inv(Kzz) @ Kzx + noise_level * I
        noise_level = tf.get_variable(
            'noise_level', shape=[], dtype=hps.dtype,
            initializer=tf.constant_initializer(0.5))
        noise_level = tf.nn.softplus(noise_level)

        # With ill-conditioned Kzz, the inverse is often asymmetric, which
        # breaks further cholesky decomposition. We compute a symmetric one.
        Kzz_chol_inv = tf.matrix_triangular_solve(
            Kzz_chol, tf.eye(hps.n_z, dtype=hps.dtype))
        Kzz_inv = tf.matmul(tf.transpose(Kzz_chol_inv), Kzz_chol_inv)

        Kxziz = tf.matmul(Kxz, Kzz_inv)
        mean_y = tf.matmul(fZ, tf.transpose(Kxziz)) # [n_particles, n_cov]

        cov_fx_given_fz = rbf(x_ph, x_ph, k_inv_scale) - \
            tf.matmul(Kxziz, tf.transpose(Kxz))
        cov_y = cov_fx_given_fz + noise_level * tf.eye(
            tf.shape(x_ph)[0], dtype=hps.dtype)
        if full_cov:
            cov_y_chol = tf.tile(tf.expand_dims(tf.cholesky(cov_y), 0),
                                 [n_particles, 1, 1])
            y = zs.MultivariateNormalCholesky('y', mean_y, cov_y_chol)
        else:
            std_y = tf.sqrt(tf.matrix_diag_part(cov_y))
            std_y = tf.tile(tf.expand_dims(std_y, 0), [n_particles, 1])
            y = zs.Normal('y', mean=mean_y, std=std_y, group_ndims=1)

    return model, y


def build_variational(hps, n_particles):
    with zs.BayesianNet() as variational:
        z_mean = tf.get_variable(
            'z/mean', [hps.n_z], hps.dtype, tf.zeros_initializer())
        z_cov_raw = tf.get_variable(
            'z/cov_raw', initializer=tf.eye(hps.n_z, dtype=hps.dtype))
        z_cov_tril = tf.matrix_set_diag(
            tf.matrix_band_part(z_cov_raw, -1, 0),
            tf.nn.softplus(tf.matrix_diag_part(z_cov_raw)))
        z = zs.MultivariateNormalCholesky(
            'fZ', z_mean, z_cov_tril, n_samples=n_particles)
    return variational


def main():
    tf.set_random_seed(1237)
    np.random.seed(1234)
    hps = parser.parse_args()

    # Load data
    data_path = os.path.join(conf.data_dir, hps.dataset + '.data')
    data_func = getattr(dataset, 'load_uci_' + hps.dataset)
    x_train, y_train, x_valid, y_valid, x_test, y_test = data_func(data_path)
    x_train = np.vstack([x_train, x_valid])
    y_train = np.hstack([y_train, y_valid])
    n_train, hps.n_covariates = x_train.shape
    hps.dtype = getattr(tf, hps.dtype)

    # Standardize data
    x_train, x_test, _, _ = dataset.standardize(x_train, x_test)
    y_train, y_test, mean_y_train, std_y_train = dataset.standardize(
        y_train, y_test)

    # Build graph
    x_ph = tf.placeholder(hps.dtype, [None, hps.n_covariates], 'x')
    y_ph = tf.placeholder(hps.dtype, [None], 'y')
    n_particles_ph = tf.placeholder(tf.int32, [], 'n_particles')
    n_batch = tf.cast(tf.shape(x_ph)[0], hps.dtype)

    # Train
    def log_joint(observed):
        model, _ = build_model(observed, x_ph, hps, n_particles_ph)
        prior, ll = model.local_log_prob(['fZ', 'y'])
        print(ll.shape)
        return prior + ll / n_batch * n_train

    variational = build_variational(hps, n_particles_ph)
    var_fZ = variational.query('fZ', outputs=True, local_log_prob=True)
    lower_bound = zs.variational.elbo(
        log_joint, observed={'y': y_ph}, latent={'fZ': var_fZ}, axis=0)
    cost = tf.reduce_mean(lower_bound.sgvb())
    lower_bound = tf.reduce_mean(lower_bound)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    infer_op = optimizer.minimize(cost)

    def infer_step(x_batch, y_batch):
        fd = {x_ph: x_batch, y_ph: y_batch, n_particles_ph: hps.n_particles}
        return sess.run([infer_op, cost], fd)[1]

    # Predict
    observed = {'fZ': var_fZ[0], 'y': y_ph}
    model, y_inferred = build_model(observed, x_ph, hps, n_particles_ph)
    log_likelihood = model.local_log_prob('y')

    def log_mean_exp(x, axis=None, keep_dims=False):
        # FIXME: Any reason zs.log_mean_exp cast to tf.float32?
        x_max = tf.reduce_max(x, axis=axis, keep_dims=True)
        ret = tf.log(tf.reduce_mean(tf.exp(x - x_max), axis=axis,
                                    keep_dims=True)) + x_max
        if not keep_dims:
            ret = tf.reduce_mean(ret, axis=axis)
        return ret

    std_y_train = tf.cast(std_y_train, hps.dtype)
    log_likelihood = log_mean_exp(log_likelihood, 0) / n_batch - tf.log(std_y_train)
    y_pred_mean = tf.reduce_mean(y_inferred.distribution.mean, axis=0)
    pred_mse = tf.reduce_mean((y_pred_mean - y_ph) ** 2)
    pred_mse *= std_y_train ** 2

    def predict_step(x_batch, y_batch):
        fd = {
            x_ph: x_batch,
            y_ph: y_batch,
            n_particles_ph: hps.n_particles_test
        }
        return sess.run([log_likelihood, pred_mse], fd)

    # Run the inference
    iters = int(np.ceil(x_train.shape[0] / float(hps.n_batch)))
    test_freq = 10
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(1, hps.n_epoch + 1):
            lbs = []
            indices = np.arange(x_train.shape[0])
            np.random.shuffle(indices)
            for t in range(iters):
                idx = indices[t * hps.n_batch: (t + 1) * hps.n_batch]
                lb = infer_step(x_train[idx], y_train[idx])
                lbs.append(lb)
            print('Epoch {}: Lower bound = {}'.format(epoch, np.mean(lbs)))
            if epoch % test_freq == 0:
                test_lls = []
                test_mses = []
                for t in range(0, x_test.shape[0], hps.n_batch):
                    ll, mse = predict_step(
                        x_test[t: t+hps.n_batch], y_test[t: t+hps.n_batch])
                    test_lls.append(ll)
                    test_mses.append(mse)
                print('>> TEST')
                print('>> Test log likelihood = {}, rmse = {}'.format(
                    np.mean(test_lls), np.sqrt(np.mean(test_mses))))


if __name__ == '__main__':
    main()
