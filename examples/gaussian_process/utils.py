'''
Gaussian Process utils. Design inspired by GPflow.
'''

import numpy as np
import tensorflow as tf
import zhusuan as zs


class RBFKernel:

    def __init__(self, n_inp, name='rbf_kernel', dtype=tf.float32):
        k_raw_scale = tf.get_variable('k_log_scale_{}'.format(name),
                                      [n_inp], dtype,
                                      initializer=tf.zeros_initializer())
        self.k_scale = tf.nn.softplus(k_raw_scale)

    def __call__(self, x, y):
        '''
        Return K(x, y), where x and y are possibly batched.
        :param x: shape [n_x, n_covariates] or [n_batch, n_x, n_covariates]
        :param y: shape [n_y, n_covariates] or [n_batch, n_y, n_covariates]
        '''
        should_squeeze = (x.shape.ndims == 2) and (y.shape.ndims == 2)
        if x.shape.ndims < 3:
            x = tf.expand_dims(x, 0)
        if y.shape.ndims < 3:
            y = tf.expand_dims(y, 0)
        x = tf.expand_dims(x, 2)
        y = tf.expand_dims(y, 1)
        inv_scale = tf.reshape(self.k_scale, [1, 1, 1, -1])
        ret = tf.exp(-tf.reduce_sum(tf.square(x - y) / inv_scale, axis=-1) / 2)
        return ret if not should_squeeze else tf.squeeze(ret, [0])

    def Kdiag(self, x):
        '''
        Optimized equivalent of diag_part(self(x, x))
        '''
        if x.shape.ndims == 2:
            return tf.ones([tf.shape(x)[0]], dtype=x.dtype)
        else:
            return tf.ones([tf.shape(x)[0], tf.shape(x)[1]], dtype=x.dtype)


def gp_conditional(Z, fZ, X, full_cov, kernel, name, Kzz_chol=None):
    '''
    GP gp_conditional f(X) | f(Z)==fZ
    :param Z: shape [n_z, n_covariates]
    :param fZ: shape [n_particles, n_z]
    :param X: shape [n_x, n_covariates]
    '''
    n_z = int(Z.shape[0])
    n_particles = tf.shape(fZ)[0]

    if Kzz_chol is None:
        Kzz_chol = tf.cholesky(kernel(Z, Z))

    # Mean[fX|fZ] = Kxz @ inv(Kzz) @ fZ; Cov[fX|Z] = Kxx - Kxz @ inv(Kzz) @ Kzx
    # With ill-conditioned Kzz, the inverse is often asymmetric, which
    # breaks further cholesky decomposition. We compute a symmetric one.
    Kzz_chol_inv = tf.matrix_triangular_solve(Kzz_chol, tf.eye(n_z))
    Kzz_inv = tf.matmul(tf.transpose(Kzz_chol_inv), Kzz_chol_inv)
    Kxz = kernel(X, Z)  # [n_x, n_z]
    Kxziz = tf.matmul(Kxz, Kzz_inv)
    mean_fx_given_fz = tf.matmul(fZ, tf.matrix_transpose(Kxziz))

    if full_cov:
        cov = kernel(X, X) - tf.matmul(Kxziz, tf.transpose(Kxz))
        cov = tf.tile(tf.expand_dims(tf.cholesky(cov), 0),
                      [n_particles, 1, 1])
        fx_given_fz = zs.MultivariateNormalCholesky(
            name, mean_fx_given_fz, cov_fx_given_fz)
    else:
        # diag(AA^T) = sum(A**2, axis=-1)
        std = kernel.Kdiag(X) - \
            tf.reduce_sum(tf.matmul(Kxz, Kzz_chol_inv) ** 2, axis=-1)
        fx_given_fz = zs.Normal(
            name, mean=mean_fx_given_fz, std=std, group_ndims=1)
    return fx_given_fz
