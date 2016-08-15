#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import sys
import os

import tensorflow as tf
import prettytensor as pt
from six.moves import range
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from zhusuan.distributions import norm, bernoulli
    from zhusuan.utils import log_mean_exp
    from zhusuan.variational import ReparameterizedNormal, advi
except:
    raise ImportError()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import dataset
    from deconv import deconv2d
except:
    raise ImportError()


class M1:
    """
    The deep generative model used in variational autoencoder (VAE).

    :param n_z: Int. The dimension of latent variables (z).
    :param n_x: Int. The dimension of observed variables (x).
    """
    def __init__(self, n_z, n_x):
        self.n_z = n_z
        self.n_x = n_x
        with pt.defaults_scope(activation_fn=tf.nn.relu,
                               scale_after_normalization=True):
            self.l_x_z = (pt.template('z').
                          reshape([-1, 1, 1, self.n_z]).
                          deconv2d(3, 128, edges='VALID').
                          batch_normalize().
                          deconv2d(5, 64, edges='VALID').
                          batch_normalize().
                          deconv2d(5, 32, stride=2).
                          batch_normalize().
                          deconv2d(5, 1, stride=2, activation_fn=tf.nn.sigmoid))

    def log_prob(self, z, x):
        """
        The joint likelihood of M1 deep generative model.

        :param z: Tensor of shape (batch_size, samples, n_z). n_z is the
            dimension of latent variables.
        :param x: Tensor of shape (batch_size, n_x). n_x is the dimension of
            observed variables (data).

        :return: A Tensor of shape (batch_size, samples). The joint log
            likelihoods.
        """

        l_x_z = self.l_x_z.construct(
            z=tf.reshape(z, (-1, self.n_z))).reshape(
            (-1, int(z.get_shape()[1]), self.n_x)).tensor
        log_px_z = tf.reduce_sum(
            bernoulli.logpdf(tf.expand_dims(x, 1), l_x_z, eps=1e-6), 2)
        log_pz = tf.reduce_sum(norm.logpdf(z), 2)
        return log_px_z + log_pz


def q_net(x, n_z, n_x=28):
    """
    Build the recognition network (Q-net) used as variational posterior.

    :param x: Tensor of shape (batch_size, n_x).
    :param n_z: Int. The dimension of latent variables (z).
    :param n_x: Int. The dimension of input image height or length
    :return: A Tensor of shape (batch_size, n_z). Variational mean of latent
        variables.
    :return: A Tensor of shape (batch_size, n_z). Variational log standard
        deviation of latent variables.
    """
    with pt.defaults_scope(activation_fn=tf.nn.relu,
                           scale_after_normalization=True):
        l_z_x = (pt.wrap(x).
                 reshape([-1, n_x, n_x, 1]).
                 conv2d(5, 32, stride=2).
                 conv2d(5, 64, stride=2).
                 batch_normalize().
                 conv2d(5, 128, edges='VALID').
                 batch_normalize().
                 dropout(0.9).
                 flatten())
        l_z_x_mean = l_z_x.fully_connected(n_z, activation_fn=None)
        l_z_x_logstd = l_z_x.fully_connected(n_z, activation_fn=None)
    return l_z_x_mean, l_z_x_logstd


def is_loglikelihood(model, x, z_proposal, n_samples=1000):
    """
    Data log likelihood (:math:`\log p(x)`) estimates using self-normalized
    importance sampling.

    :param model: A model object that has a method logprob(z, x) to compute the
        log joint likelihood of the model.
    :param x: A Tensor of shape (batch_size, n_x). The observed variables (
        data).
    :param z_proposal: A :class:`Variational` object used as the proposal
        in importance sampling.
    :param n_samples: Int. Number of samples used in this estimate.

    :return: A Tensor of shape (batch_size,). The log likelihood of data (x).
    """
    samples = z_proposal.sample(n_samples)
    log_w = model.log_prob(samples, x) - z_proposal.logpdf(samples)
    return log_mean_exp(log_w, 1)


if __name__ == "__main__":
    tf.set_random_seed(1234)

    # Load MNIST
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'data', 'mnist.pkl.gz')
    x_train, t_train, x_valid, t_valid, x_test, t_test = \
        dataset.load_mnist_realval(data_path)
    x_train = np.vstack([x_train, x_valid])
    np.random.seed(1234)
    x_test = np.random.binomial(1, x_test, size=x_test.shape).astype('float32')

    # Define hyper-parameters
    n_z = 40

    # Define training/evaluation parameters
    lb_samples = 1
    ll_samples = 100
    epoches = 3000
    batch_size = 100
    test_batch_size = 100
    iters = x_train.shape[0] // batch_size
    test_iters = x_test.shape[0] // test_batch_size
    test_freq = 10

    # Build the training computation graph
    x = tf.placeholder(tf.float32, shape=(batch_size, x_train.shape[1]))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001, epsilon=1e-4)
    with pt.defaults_scope(phase=pt.Phase.train):
        with tf.variable_scope("model") as scope:
            train_model = M1(n_z, x_train.shape[1])
        with tf.variable_scope("variational") as scope:
            train_vz_mean, train_vz_logstd = q_net(x, n_z)
            train_variational = ReparameterizedNormal(
                train_vz_mean, train_vz_logstd)
    grads, lower_bound = advi(
        train_model, x, train_variational, lb_samples, optimizer)
    infer = optimizer.apply_gradients(grads)

    # Build the evaluation computation graph
    with pt.defaults_scope(phase=pt.Phase.test):
        with tf.variable_scope("model", reuse=True) as scope:
            eval_model = M1(n_z, x_train.shape[1])
        with tf.variable_scope("variational", reuse=True) as scope:
            eval_vz_mean, eval_vz_logstd = q_net(x, n_z)
            eval_variational = ReparameterizedNormal(
                eval_vz_mean, eval_vz_logstd)
    eval_lower_bound = is_loglikelihood(
        eval_model, x, eval_variational, lb_samples)
    eval_log_likelihood = is_loglikelihood(
        eval_model, x, eval_variational, ll_samples)

    params = tf.trainable_variables()
    for i in params:
        print(i.name, i.get_shape())

    init = tf.initialize_all_variables()

    # Run the inference
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(1, epoches + 1):
            np.random.shuffle(x_train)
            lbs = []
            for t in range(iters):
                x_batch = x_train[t * batch_size:(t + 1) * batch_size]
                x_batch = np.random.binomial(
                    n=1, p=x_batch, size=x_batch.shape).astype('float32')
                _, lb = sess.run([infer, lower_bound], feed_dict={x: x_batch})
                lbs.append(lb)
            print('Epoch {}: Lower bound = {}'.format(epoch, np.mean(lbs)))
            if epoch % test_freq == 0:
                test_lbs = []
                test_lls = []
                for t in range(test_iters):
                    test_x_batch = x_test[
                        t * test_batch_size: (t + 1) * test_batch_size]
                    test_lb, test_ll = sess.run(
                        [eval_lower_bound, eval_log_likelihood],
                        feed_dict={x: test_x_batch}
                    )
                    test_lbs.append(test_lb)
                    test_lls.append(test_ll)
                print('>> Test lower bound = {}'.format(np.mean(test_lbs)))
                print('>> Test log likelihood = {}'.format(np.mean(test_lls)))
