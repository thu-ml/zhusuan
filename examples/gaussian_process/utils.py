'''
Gaussian Process utils. Design inspired by GPflow.
'''

import numpy as np
import tensorflow as tf
import zhusuan as zs


class RBFKernel:

    def __init__(self, n_covariates, name='rbf_kernel', dtype=tf.float32):
        k_raw_scale = tf.get_variable('k_log_scale_{}'.format(name),
                                      [n_covariates], dtype,
                                      initializer=tf.zeros_initializer())
        self.k_scale = tf.nn.softplus(k_raw_scale)

    def __call__(self, x, y):
        '''
        Return K(x, y), where x and y are possibly batched.
        :param x: shape [..., n_x, n_covariates]
        :param y: shape [..., n_y, n_covariates]
        :return: Tensor with shape [..., n_x, n_y]
        '''
        batch_shape = tf.shape(x)[:-2]
        rank = x.shape.ndims
        assert_ops = [
            tf.assert_greater_equal(
                rank, 2,
                message='RBFKernel: rank(x) should be static and >=2'),
            tf.assert_equal(
                rank, tf.rank(y),
                message='RBFKernel: x and y should have the same rank')]
        with tf.control_dependencies(assert_ops):
            x = tf.expand_dims(x, rank - 1)
            y = tf.expand_dims(y, rank - 2)
            k_scale = tf.reshape(self.k_scale, [1] * rank + [-1])
            ret = tf.exp(
                -tf.reduce_sum(tf.square(x - y) / k_scale, axis=-1) / 2)
        return ret

    def Kdiag(self, x):
        '''
        Optimized equivalent of diag_part(self(x, x))
        '''
        if x.shape.ndims == 2:
            return tf.ones([tf.shape(x)[0]], dtype=x.dtype)
        else:
            return tf.ones([tf.shape(x)[0], tf.shape(x)[1]], dtype=x.dtype)


def gp_conditional(z, fz, x, full_cov, kernel, name, Kzz_chol=None):
    '''
    GP gp_conditional f(x) | f(z)==fz
    :param z: shape [n_z, n_covariates]
    :param fz: shape [n_particles, n_z]
    :param x: shape [n_x, n_covariates]
    :return: StochasticTensor with shape [n_particles, n_x]
    '''
    n_z = int(z.shape[0])
    n_particles = tf.shape(fz)[0]

    if Kzz_chol is None:
        Kzz_chol = tf.cholesky(kernel(z, z))

    # Mean[fx|fz] = Kxz @ inv(Kzz) @ fz; Cov[fx|z] = Kxx - Kxz @ inv(Kzz) @ Kzx
    # With ill-conditioned Kzz, the inverse is often asymmetric, which
    # breaks further cholesky decomposition. We compute a symmetric one.
    Kzz_chol_inv = tf.matrix_triangular_solve(Kzz_chol, tf.eye(n_z))
    Kzz_inv = tf.matmul(tf.transpose(Kzz_chol_inv), Kzz_chol_inv)
    Kxz = kernel(x, z)  # [n_x, n_z]
    Kxziz = tf.matmul(Kxz, Kzz_inv)
    mean_fx_given_fz = tf.matmul(fz, tf.matrix_transpose(Kxziz))

    if full_cov:
        cov_fx_given_fz = kernel(x, x) - tf.matmul(Kxziz, tf.transpose(Kxz))
        cov_fx_given_fz = tf.tile(
            tf.expand_dims(tf.cholesky(cov_fx_given_fz), 0),
            [n_particles, 1, 1])
        fx_given_fz = zs.MultivariateNormalCholesky(
            name, mean_fx_given_fz, cov_fx_given_fz)
    else:
        # diag(AA^T) = sum(A**2, axis=-1)
        var = kernel.Kdiag(x) - \
            tf.reduce_sum(tf.matmul(
                Kxz, tf.matrix_transpose(Kzz_chol_inv)) ** 2, axis=-1)
        std = tf.sqrt(var)
        fx_given_fz = zs.Normal(
            name, mean=mean_fx_given_fz, std=std, group_ndims=1)
    return fx_given_fz
