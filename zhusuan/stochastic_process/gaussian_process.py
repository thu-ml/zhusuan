#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf

from zhusuan.distributions.univariate import Normal
from zhusuan.distributions.multivariate import MultivariateNormalCholesky
from zhusuan.stochastic_process.base import StochasticProcess

__all__ = [
    'GaussianProcess'
]


class GaussianProcess(StochasticProcess):
    def __init__(self,
                 mean_function,
                 kernel):
        super(GaussianProcess, self).__init__()
        # TODO: check if number of mean function is equal to number of kernel
        self.mean_function = mean_function
        self.kernel = kernel

    def instantiate(self,
                    positions,
                    noise_level=0,
                    group_ndims=0,
                    is_reparameterized=True,
                    use_path_derivative=False,
                    check_numerics=False,
                    **kwargs):
        rank = positions.shape.ndims
        mean = self.mean_function(positions)
        batch_size = tf.shape(positions)[-2]
        if rank == 2:
            cov = self.kernel(positions, positions) + noise_level * tf.tile(
                tf.reshape(tf.eye(batch_size), [1, batch_size, batch_size]), [self.kernel.kernel_num, 1, 1])
        if rank == 3:
            cov = self.kernel(positions, positions) + noise_level * tf.tile(
                tf.reshape(tf.eye(batch_size), [1, 1, batch_size, batch_size]),
                [int(positions.shape[0]), self.kernel.kernel_num, 1, 1])
        cov_cholesky = tf.cholesky(cov)
        return MultivariateNormalCholesky(mean, cov_cholesky, group_ndims=group_ndims,
                                          is_reparameterized=is_reparameterized,
                                          use_path_derivative=use_path_derivative, check_numerics=check_numerics)

    def conditional(self,
                    x,
                    inducing_points,
                    inducing_values,
                    factorized=False,
                    noise_level=0,
                    group_ndims=0,
                    is_reparameterized=True,
                    use_path_derivative=False,
                    check_numerics=False,
                    **kwargs):
        assert_ops = [tf.assert_equal(x.shape[-1], self.kernel.covariates_num,
                                      message='GaussianProcess: inducing points\' covariates number should match the kernel'),
                      tf.assert_equal(x.shape[-1], inducing_points.shape[-1],
                                      message='GaussianProcess: covariates number should be same between inducing points and data points')]
        n_particles = tf.shape(inducing_points)[0]
        batch_size = tf.shape(inducing_points)[-2]
        with tf.control_dependencies(assert_ops):
            cov_z = self.kernel(inducing_points, inducing_points)
            cov_z = cov_z + noise_level * tf.tile(tf.reshape(tf.eye(batch_size), [1, 1, batch_size, batch_size]),
                                                  [n_particles, self.kernel.kernel_num, 1, 1])
            # For positive-semidefinite convariance matrix
            cov_z_cholesky = tf.cholesky(cov_z)
            cov_z_cholesky_inverse = tf.matrix_triangular_solve(cov_z_cholesky, tf.tile(
                tf.reshape(tf.eye(int(cov_z_cholesky.shape[-1])),
                           [1, 1, int(cov_z_cholesky.shape[-1]), int(cov_z_cholesky.shape[-1])]),
                [n_particles, self.kernel.kernel_num, 1, 1]))
            cov_z_inverse = tf.matmul(cov_z_cholesky_inverse,
                                      cov_z_cholesky_inverse,
                                      transpose_a=True)
        cov_z_x = self.kernel(inducing_points, x)
        mean = self.mean_function(x) + tf.reduce_sum(
            tf.multiply(tf.expand_dims(inducing_values - self.mean_function(inducing_points), -1),
                        tf.matmul(cov_z_inverse, cov_z_x)), -2)
        if factorized:
            std = tf.sqrt(self.kernel.Kdiag(x) - tf.reduce_sum(
                tf.matmul(cov_z_x, tf.matrix_transpose(cov_z_cholesky_inverse), transpose_a=True) ** 2,
                axis=-1))
            return Normal(mean, std=std, group_ndims=group_ndims, is_reparameterized=is_reparameterized,
                          use_path_derivative=use_path_derivative, check_numerics=check_numerics)

        else:
            cov_x_values = self.kernel(x, x) - tf.matmul(tf.matmul(cov_z_x, cov_z_inverse, transpose_a=True),
                                                         cov_z_x)
            cov_x_values_cholesky = tf.cholesky(cov_x_values)
            return MultivariateNormalCholesky(mean, cov_x_values_cholesky, group_ndims=group_ndims,
                                              is_reparameterized=is_reparameterized,
                                              use_path_derivative=use_path_derivative, check_numerics=check_numerics)