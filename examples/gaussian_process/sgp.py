#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Sparse variational Gaussian process, loosely based on

> Opper and Archambeau 2009, The variational Gaussian approximation revisited.

On the Boston dataset you should get ~2.7 RMSE and ~2.3 negative log likelihood
after a while with default args;
On the protein dataset you should get ~4.4 RMSE and ~2.9 NLL after 70 epochs
with `N_batch=1000 N_particles=10 dtype=float32`.
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
parser.add_argument('-N_Z', default=100, type=int)
parser.add_argument('-N_X', default=1, type=int)
parser.add_argument('-N_particles', default=50, type=int)
parser.add_argument('-N_particles_test', default=100, type=int)
parser.add_argument('-N_batch', default=100, type=int)
parser.add_argument('-N_epoch', default=5000, type=int)
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
def build_model(observed, Z_pos, X_ph, hps, n_particles):
    with zs.BayesianNet(observed) as model:
        k_inv_scale = tf.get_variable('k_inv_scale', [hps.N_X], hps.dtype,
                                      tf.zeros_initializer())
        k_inv_scale = tf.exp(k_inv_scale)
        Kzz = rbf(Z_pos, Z_pos, k_inv_scale)
        Kxz = rbf(X_ph, Z_pos, k_inv_scale)
        Kzz_chol = tf.cholesky(Kzz)
        fZ = zs.MultivariateNormalCholesky(
            'fZ', tf.zeros([hps.N_Z], dtype=hps.dtype), Kzz_chol,
            n_samples=n_particles)
        # Mean = Kxz @ inv(Kzz) @ fZ
        # When Z works Kzz will have small eigenvalues. Take care.
        Kzz_chol_inv = tf.matrix_triangular_solve(
            Kzz_chol, tf.eye(hps.N_Z, dtype=hps.dtype))
        Kxziz = tf.matmul(tf.matmul(Kxz, tf.transpose(Kzz_chol_inv)),
                          Kzz_chol_inv)
        mean_xz = tf.matmul(fZ, tf.transpose(Kxziz))  # [N_particles, N_X]
        cov_xz = rbf(X_ph, X_ph, k_inv_scale) -\
            tf.matmul(Kxziz, tf.transpose(Kxz))
        noise_level = tf.get_variable(
            'noise_level', dtype=hps.dtype,
            initializer=tf.constant(0.5, dtype=hps.dtype))
        noise_level = tf.nn.softplus(noise_level)
        cov_y = cov_xz + noise_level * tf.eye(
            tf.shape(X_ph)[0], dtype=hps.dtype)
        cov_y_chol = tf.tile(tf.expand_dims(tf.cholesky(cov_y), 0),
                             [n_particles, 1, 1])
        Y = zs.MultivariateNormalCholesky('Y', mean_xz, cov_y_chol)
    return model, Y


def build_variational(hps, n_particles):
    with zs.BayesianNet() as variational:
        Z_mean = tf.get_variable(
            'Z/mean', [hps.N_Z], hps.dtype, tf.zeros_initializer())
        Z_cov_raw = tf.get_variable(
            'Z/cov_raw', initializer=tf.eye(hps.N_Z, dtype=hps.dtype))
        Z_cov_tril = tf.matrix_set_diag(
            tf.matrix_band_part(Z_cov_raw, -1, 0),
            tf.nn.softplus(tf.matrix_diag_part(Z_cov_raw)))
        Z = zs.MultivariateNormalCholesky(
            'fZ', Z_mean, Z_cov_tril, n_samples=n_particles)
    return variational


def main():
    tf.set_random_seed(1237)
    np.random.seed(1234)
    hps = parser.parse_args()

    # Load UCI Boston housing data
    data_path = os.path.join(conf.data_dir, hps.dataset + '.data')
    data_func = getattr(dataset, 'load_uci_' + hps.dataset)
    x_train, y_train, x_valid, y_valid, x_test, y_test = data_func(data_path)
    x_train = np.vstack([x_train, x_valid])
    y_train = np.hstack([y_train, y_valid])
    N, hps.N_X = x_train.shape
    hps.dtype = getattr(tf, hps.dtype)

    # Standardize data
    x_train, x_test, _, _ = dataset.standardize(x_train, x_test)
    y_train, y_test, mean_y_train, std_y_train = dataset.standardize(
        y_train, y_test)

    # Build graph
    X_ph = tf.placeholder(hps.dtype, [None, hps.N_X], 'X')
    Y_ph = tf.placeholder(hps.dtype, [None], 'Y')
    Z_pos = tf.get_variable('Z/pos', [hps.N_Z, hps.N_X], hps.dtype,
                            initializer=tf.random_uniform_initializer(-1, 1))
    N_particles = tf.placeholder(tf.int32, [], 'n_particles')

    # Train
    def log_joint(observed):
        model, _ = build_model(observed, Z_pos, X_ph, hps, N_particles)
        prior, ll = model.local_log_prob(['fZ', 'Y'])
        n_batch = tf.cast(tf.shape(X_ph)[0], hps.dtype)
        return prior + ll / n_batch * N

    variational = build_variational(hps, N_particles)
    var_fZ = variational.query('fZ', outputs=True, local_log_prob=True)
    lower_bound = zs.variational.elbo(
        log_joint, observed={'Y': Y_ph}, latent={'fZ': var_fZ}, axis=0)
    cost = tf.reduce_mean(lower_bound.sgvb())
    lower_bound = tf.reduce_mean(lower_bound)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    infer_op = optimizer.minimize(cost)

    def infer_step(X_batch, Y_batch):
        fd = {X_ph: X_batch, Y_ph: Y_batch, N_particles: hps.N_particles}
        return sess.run([infer_op, cost], fd)[1]

    # Predict
    observed = {'fZ': var_fZ[0], 'Y': Y_ph}
    model, y_inferred = build_model(observed, Z_pos, X_ph, hps, N_particles)
    lhood = model.local_log_prob('Y')
    N_Y = tf.cast(tf.shape(Y_ph)[0], hps.dtype)

    def log_mean_exp(x, axis=None, keep_dims=False):
        # FIXME: Any reason zs.log_mean_exp cast to tf.float32?
        x_max = tf.reduce_max(x, axis=axis, keep_dims=True)
        ret = tf.log(tf.reduce_mean(tf.exp(x - x_max), axis=axis,
                                    keep_dims=True)) + x_max
        if not keep_dims:
            ret = tf.reduce_mean(ret, axis=axis)
        return ret

    std_y_train = tf.cast(std_y_train, hps.dtype)
    lhood = log_mean_exp(lhood, 0) / N_Y - tf.log(std_y_train)
    y_pred_mean = tf.reduce_mean(y_inferred.distribution.mean, axis=0)
    pred_mse = tf.reduce_mean((y_pred_mean - Y_ph) ** 2)
    pred_mse *= std_y_train ** 2

    def predict_step(X_batch, Y_batch):
        fd = {X_ph: X_batch, Y_ph: Y_batch, N_particles: hps.N_particles_test}
        return sess.run([lhood, pred_mse], fd)

    # Run the inference
    iters = int(np.floor(x_train.shape[0] / float(hps.N_batch)))
    test_freq = 10
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(1, hps.N_epoch + 1):
            lbs = []
            indices = np.arange(x_train.shape[0])
            np.random.shuffle(indices)
            for t in range(iters):
                idx = indices[t * hps.N_batch: (t + 1) * hps.N_batch]
                lb = infer_step(x_train[idx], y_train[idx])
                lbs.append(lb)
            print('Epoch {}: Lower bound = {}'.format(epoch, np.mean(lbs)))
            if epoch % test_freq == 0:
                test_lls = []
                test_mses = []
                for t in range(0, x_test.shape[0], hps.N_batch):
                    ll, mse = predict_step(
                        x_test[t: t+hps.N_batch], y_test[t: t+hps.N_batch])
                    test_lls.append(ll)
                    test_mses.append(mse)
                print('>> TEST')
                print('>> Test lhood = {}, rmse = {}'.format(
                    np.mean(test_lls), np.sqrt(np.mean(test_mses))))


if __name__ == '__main__':
    main()
