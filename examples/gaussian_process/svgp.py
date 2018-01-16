#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Sparse variational Gaussian process with normal approximation for posterior.

For the formulation you can refer to, e.g., Section 2.1 of the following paper:

Salimbeni and Deisenroth 2017, Doubly Stochastic Variational Inference for Deep
Gaussian Processes.

Results (RMSE, NLL):
 N_Z   Boston    Protein
----- --------- ---------
 100  2.73,2.50
 500            4.08,2.86
'''

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from six.moves import range, zip

import numpy as np
import tensorflow as tf
import zhusuan as zs
from zhusuan.utils import log_mean_exp
from examples import conf
from examples.utils import dataset
import os
import argparse
from utils import gp_conditional, RBFKernel


parser = argparse.ArgumentParser()
parser.add_argument('-n_z', default=100, type=int)
parser.add_argument('-n_covariates', default=1, type=int)
parser.add_argument('-n_particles', default=20, type=int)
parser.add_argument('-n_particles_test', default=100, type=int)
parser.add_argument('-n_batch', default=5000, type=int)
parser.add_argument('-n_epoch', default=5000, type=int)
parser.add_argument('-dtype', default='float32', type=str,
                    choices=['float32', 'float64'])
parser.add_argument('-dataset', default='boston_housing', type=str,
                    choices=['boston_housing', 'protein_data'])


class SVGP:

    def __init__(self, hps, n_train, std_y_train):
        self._hps = hps
        self._kernel = RBFKernel(hps.n_covariates)
        self._x_ph = x_ph = tf.placeholder(
            hps.dtype, [None, hps.n_covariates], 'x')
        self._y_ph = y_ph = tf.placeholder(
            hps.dtype, [None], 'y')
        self._z_pos = tf.get_variable(
            'z/pos', [hps.n_z, hps.n_covariates], hps.dtype,
            initializer=tf.random_uniform_initializer(-1, 1))
        self._n_particles_ph = n_particles_ph = tf.placeholder(
            tf.int32, [], 'n_particles')
        n_batch = tf.cast(tf.shape(x_ph)[0], hps.dtype)

        # TRAIN
        # ELBO = E_q log (p(y|fx)p(fx|fz)p(fz) / p(fx|fz)q(fz))
        # So we remove p(fx|fz) in both log_joint and latent
        def log_joint(observed):
            model, _ = SVGP.build_model(self, observed, n_particles_ph)
            prior, log_p_y_given_fx = model.local_log_prob(['fZ', 'y'])
            return prior + log_p_y_given_fx / n_batch * n_train

        variational = self.build_variational(n_particles_ph)
        [var_fZ, var_fX] = variational.query(
            ['fZ', 'fX'], outputs=True, local_log_prob=True)
        var_fX = (var_fX[0], tf.zeros_like(var_fX[1]))
        lower_bound = zs.variational.elbo(
            log_joint,
            observed={'y': y_ph},
            latent={'fZ': var_fZ, 'fX': var_fX},
            axis=0)
        self._elbo = lower_bound.sgvb()
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        self._infer_op = optimizer.minimize(self._elbo)

        # PREDICT
        observed = {'fX': var_fX[0], 'y': y_ph}
        model, y_inferred = SVGP.build_model(self, observed, n_particles_ph)
        log_likelihood = model.local_log_prob('y')
        std_y_train = tf.cast(std_y_train, hps.dtype)
        self.log_likelihood = log_mean_exp(log_likelihood, 0) / n_batch - \
            tf.log(std_y_train)
        y_pred_mean = tf.reduce_mean(y_inferred.distribution.mean, axis=0)
        self.pred_mse = tf.reduce_mean((y_pred_mean - y_ph) ** 2)
        self.pred_mse *= std_y_train ** 2

    @zs.reuse('model')
    def build_model(self, observed, n_particles, full_cov=False):
        '''
        Build the SVGP model.
        Note that for inference, we only need the diagonal part of Cov[Y], as
        ELBO equals sum over individual observations.
        For visualization etc we may want a full covariance. Thus the argument
        `full_cov`.
        '''
        hps = self._hps
        with zs.BayesianNet(observed) as model:
            Kzz_chol = tf.cholesky(self._kernel(self._z_pos, self._z_pos))
            fZ = zs.MultivariateNormalCholesky(
                'fZ', tf.zeros([hps.n_z], dtype=hps.dtype), Kzz_chol,
                n_samples=n_particles)
            # f(X)|f(Z) follows GP(0, K) gp_conditional
            fx_given_fz = gp_conditional(
                self._z_pos, fZ, self._x_ph, full_cov, self._kernel, 'fX',
                Kzz_chol)
            # Y|f(X) ~ N(f(X), noise_level * I)
            noise_level = tf.get_variable(
                'noise_level', shape=[], dtype=hps.dtype,
                initializer=tf.constant_initializer(0.05))
            noise_level = tf.nn.softplus(noise_level)
            y = zs.Normal(
                'y', mean=fx_given_fz, std=noise_level, group_ndims=1)
        return model, y

    def build_variational(self, n_particles):
        hps = self._hps
        with zs.BayesianNet() as variational:
            z_mean = tf.get_variable(
                'z/mean', [hps.n_z], hps.dtype, tf.zeros_initializer())
            z_cov_raw = tf.get_variable(
                'z/cov_raw', initializer=tf.eye(hps.n_z, dtype=hps.dtype))
            z_cov_tril = tf.matrix_set_diag(
                tf.matrix_band_part(z_cov_raw, -1, 0),
                tf.nn.softplus(tf.matrix_diag_part(z_cov_raw)))
            fZ = zs.MultivariateNormalCholesky(
                'fZ', z_mean, z_cov_tril, n_samples=n_particles)
            fx_given_fz = gp_conditional(
                self._z_pos, fZ, self._x_ph, False, self._kernel, 'fX')
        return variational

    def infer_step(self, sess, x_batch, y_batch):
        fd = {
            self._x_ph: x_batch,
            self._y_ph: y_batch,
            self._n_particles_ph: self._hps.n_particles
        }
        return sess.run([self._infer_op, self._elbo], fd)[1]

    def predict_step(self, sess, x_batch, y_batch):
        fd = {
            self._x_ph: x_batch,
            self._y_ph: y_batch,
            self._n_particles_ph: self._hps.n_particles_test
        }
        return sess.run([self.log_likelihood, self.pred_mse], fd)


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

    model = SVGP(hps, n_train, std_y_train)

    iters = int(np.ceil(x_train.shape[0] / float(hps.n_batch)))
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
                lb = model.infer_step(
                    sess,
                    x_train[t*hps.n_batch: (t+1)*hps.n_batch],
                    y_train[t*hps.n_batch: (t+1)*hps.n_batch])
                lbs.append(lb)
            if 10 * epoch % test_freq == 0:
                print('Epoch {}: Lower bound = {}'.format(epoch, np.mean(lbs)))
            if epoch % test_freq == 0:
                test_lls = []
                test_mses = []
                for t in range(0, x_test.shape[0], hps.n_batch):
                    ll, mse = model.predict_step(
                        sess,
                        x_test[t: t+hps.n_batch],
                        y_test[t: t+hps.n_batch])
                    test_lls.append(ll)
                    test_mses.append(mse)
                print('>> TEST')
                print('>> Test log likelihood = {}, rmse = {}'.format(
                    np.mean(test_lls), np.sqrt(np.mean(test_mses))))


if __name__ == '__main__':
    main()
