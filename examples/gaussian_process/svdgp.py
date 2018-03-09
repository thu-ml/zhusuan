#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Stochastic sparse variational Gaussian process (SVGP) with normal approximation
for posterior.

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
import os
import sys

sys.path.append('..')
sys.path.append('../..')
import argparse

import numpy as np
from six.moves import range, zip
import tensorflow as tf
import zhusuan as zs

from examples import conf
from examples.utils import dataset

parser = argparse.ArgumentParser()
parser.add_argument('-n_z', default=100, type=int)
parser.add_argument('-n_covariates', default=1, type=int)
parser.add_argument('-n_particles', default=20, type=int)
parser.add_argument('-n_particles_test', default=100, type=int)
parser.add_argument('-batch_size', default=10000, type=int)
parser.add_argument('-n_epoch', default=100000, type=int)
parser.add_argument('-dtype', default='float32', type=str,
                    choices=['float32', 'float64'])
parser.add_argument('-dataset', default='boston_housing', type=str,
                    choices=['boston_housing', 'protein_data'])


def inducing_prior(kernel, inducing_points, n_particles, name):
    cov_z_cholesky = tf.cholesky(kernel(inducing_points, inducing_points))
    return tf.tile(tf.expand_dims(inducing_points, 0), [n_particles, 1, 1]), \
           zs.MultivariateNormalCholesky('u_' + name, tf.zeros([kernel.kernel_num, int(inducing_points.shape[-2])],
                                                               dtype=tf.float32),
                                         cov_z_cholesky, n_samples=n_particles, group_ndims=1)


def variational_prior(kernel, inducing_points, n_particles, name, inner_layer=False):
    inducing_num = int(inducing_points.shape[-2])
    u_mean = tf.get_variable(
        name + '/variational_u_mean', [kernel.kernel_num, inducing_num], tf.float32, tf.zeros_initializer())
    u_cov = tf.get_variable(
        name + '/variational_u_cov',
        initializer=tf.tile(tf.expand_dims(tf.eye(inducing_num, dtype=tf.float32), 0), [kernel.kernel_num, 1, 1]))
    u_cov_tril = tf.matrix_set_diag(
        tf.matrix_band_part(u_cov, -1, 0),
        tf.matrix_diag_part(u_cov))
    if inner_layer:
        u_cov_tril *= 1e-5
    return tf.tile(tf.expand_dims(inducing_points, 0), [n_particles, 1, 1]), \
           zs.MultivariateNormalCholesky('u_' + name, u_mean, u_cov_tril, n_samples=n_particles,
                                         group_ndims=1)


@zs.reuse('model')
def build_model(observed, x, kernels, inducing_points, n_particles, factorized=True):
    '''
    Build the SVGP model.
    Note that for inference, we only need the diagonal part of Cov[Y], as
    ELBO equals sum over individual observations.
    For visualization etc we may want a full covariance. Thus the argument
    `full_cov`.
    '''
    with zs.BayesianNet(observed) as model:
        msg = 'Inducing points should match kernels!'
        assert len(kernels) == len(inducing_points), msg
        h = x
        for kernel, inducing_point, num in list(zip(kernels, inducing_points, list(range(len(kernels))))):
            assert int(inducing_point.shape[-1]) == kernel.covariates_num, msg
            noise = 1e-5
            if num == len(kernels) - 1:
                mean_function = zs.stochastic_process.ConstantMeanFunction(1, 0.)
            else:
                mean_function = zs.stochastic_process.LinearMeanFunction(kernel.covariates_num,
                                                                         np.eye(kernel.covariates_num))
            z, u = inducing_prior(kernel, inducing_point, n_particles, name='cond_' + str(num))
            h = zs.GaussianProcess('f_' + str(num), mean_function, kernel, x=h, inducing_points=z,
                                   inducing_values=u, factorized=factorized, noise_level=noise, group_ndims=2)
            h = tf.matrix_transpose(h)
        f1_given_x = tf.squeeze(h, -1)
        noise_level = tf.get_variable(
            'noise_level', shape=[], dtype=tf.float32,
            initializer=tf.constant_initializer(0.05))
        noise_level = tf.nn.softplus(noise_level)
        y = zs.Normal('y', mean=f1_given_x, std=noise_level, group_ndims=1)
    return model, y


@zs.reuse('variational')
def build_variational(x, kernels, inducing_points, n_particles, factorized=True):
    with zs.BayesianNet() as variational:
        msg = 'Inducing points should match kernels!'
        assert len(kernels) == len(inducing_points), msg
        h = x
        for kernel, inducing_point, num in list(zip(kernels, inducing_points, list(range(len(kernels))))):
            assert int(inducing_point.shape[-1]) == kernel.covariates_num, msg
            if num < len(kernels) - 1:
                inner_layer = True
                mean_function = zs.stochastic_process.LinearMeanFunction(kernel.covariates_num,
                                                                         np.eye(kernel.covariates_num))
            else:
                inner_layer = False
                mean_function = zs.stochastic_process.ConstantMeanFunction(1, 0.)
            noise = 1e-5
            z, u = variational_prior(kernel, inducing_point, n_particles, name='cond_' + str(num),
                                     inner_layer=inner_layer)
            h = zs.GaussianProcess('f_' + str(num), mean_function, kernel, x=h, inducing_points=z,
                                   inducing_values=u, factorized=factorized, noise_level=noise, group_ndims=2)
            h = tf.matrix_transpose(h)
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

    # Build model
    x_ph = tf.placeholder(hps.dtype, [None, hps.n_covariates], 'x')
    y_ph = tf.placeholder(hps.dtype, [None], 'y')
    n_particles_ph = n_particles_ph = tf.placeholder(
        tf.int32, [], 'n_particles')
    batch_size = tf.cast(tf.shape(x_ph)[0], hps.dtype)

    x_obs = tf.tile(tf.expand_dims(x_ph, 0), [n_particles_ph, 1, 1])
    y_obs = tf.tile(tf.expand_dims(y_ph, 0), [n_particles_ph, 1])
    kernels = [zs.kernels.RBFKernel(hps.n_covariates, hps.n_covariates, name='RBFkernel_0'),
               zs.kernels.RBFKernel(1, hps.n_covariates, name='RBFkernel_1')]
    inducing_points = [tf.get_variable('z_0', [hps.n_z, hps.n_covariates], hps.dtype,
                                       initializer=tf.random_uniform_initializer(-1, 1)),
                       tf.get_variable('z_1', [hps.n_z, hps.n_covariates], hps.dtype,
                                       initializer=tf.random_uniform_initializer(-1, 1))]
    # Training ops
    # ELBO = E_q log (p(y|fx)p(fx|fz)p(fz) / p(fx|fz)q(fz))
    # So we remove p(fx|fz) in both log_joint and latent
    u_name = ['u_cond_' + str(i) for i in list(range(2))]
    f_name = ['f_' + str(i) for i in list(range(2))]

    def log_joint(observed):
        model, _ = build_model(observed, x_obs, kernels, inducing_points, n_particles_ph)
        prior = model.local_log_prob(u_name)
        log_py_given_fx = model.local_log_prob(['y'])
        return tf.add_n(prior) + log_py_given_fx / batch_size * n_train

    variational = build_variational(x_obs, kernels, inducing_points, n_particles_ph)
    var_u = variational.query(u_name, outputs=True, local_log_prob=True)
    var_f = variational.query(f_name, outputs=True, local_log_prob=True)
    var_f = [(samples, tf.zeros_like(densities)) for (samples, densities) in var_f]
    # var_f_0 = (var_f_0[0], tf.zeros_like(var_f_0[1]))
    # var_f_1 = (var_f_1[0], tf.zeros_like(var_f_1[1]))
    # var_f_2 = (var_f_2[0], tf.zeros_like(var_f_2[1]))
    latent = dict(zip(u_name + f_name, var_u + var_f))
    lower_bound = zs.variational.elbo(
        log_joint,
        observed={'y': y_obs},
        latent=latent,
        axis=0)
    cost = lower_bound.sgvb()
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    print(var_list)
    grads = optimizer.compute_gradients(cost)
    infer_op = optimizer.apply_gradients(grads)

    # Prediction ops
    observed = {'f_2': var_f[-1][0], 'y': y_obs}
    model, y_inferred = build_model(observed, x_obs, kernels, inducing_points, n_particles_ph)
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
    test_freq = 10
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
                # grad = sess.run(grads, feed_dict={x_ph: x_train[t * hps.batch_size: (t + 1) * hps.batch_size],
                #                                   y_ph: y_train[t * hps.batch_size: (t + 1) * hps.batch_size],
                #                                   n_particles_ph: hps.n_particles})
                # print(list(zip(grads, grad)))
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
