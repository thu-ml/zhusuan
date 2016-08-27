#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import sys
import os
import time

import tensorflow as tf
import prettytensor as pt
from six.moves import range
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from zhusuan.distributions import norm, bernoulli
    from zhusuan.utils import log_mean_exp
    from zhusuan.layers import *
    from zhusuan.variational import advi
except:
    raise ImportError()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import dataset
except:
    raise ImportError()


class M2(object):
    """
    The deep generative model used in semi-supervised learning with variational
    autoencoders (Kingma, 2014).

    :param n_x: Int. The dimension of observed variables (x).
    :param n_y: Int. The dimension of latent variables (y), i.e., the number of
        classes.
    :param n_z: Int. The dimension of latent variables (z).
    """
    def __init__(self, n_x, n_y, n_z):
        self.n_y = n_y
        self.n_z = n_z
        self.n_x = n_x
        self.l_x_zy = self.p_net()

    def p_net(self):
        with pt.defaults_scope(activation_fn=tf.nn.relu):
            l_x_zy = (pt.template('z').join([pt.template('y')]).
                      fully_connected(500).
                      fully_connected(500).
                      fully_connected(self.n_x, activation_fn=tf.nn.sigmoid))
        return l_x_zy

    def log_prob(self, latent, observed):
        """
        The log joint probability function.

        :param latent: A dictionary of pairs: (string, Tensor). Each of the
            Tensor has shape (batch_size, n_samples, n_latent).
        :param observed: A dictionary of pairs: (string, Tensor). Each of the
            Tensor has shape (batch_size, n_observed).

        :return: A Tensor of shape (batch_size, n_samples). The joint log
            likelihoods.
        """
        raise NotImplementedError()


class M2Unlabeled(M2):
    """
    The M2 model with labels `y` as latent variables.
    """
    def __init__(self, n_x, n_y, n_z):
        super(M2Unlabeled, self).__init__(n_x, n_y, n_z)

    def log_prob(self, latent, observed):
        # y: (batch_size, 1, n_y), z: (batch_size, n_samples, n_z)
        y, z = latent['y'], latent['z']
        # x: (batch_size, n_x)
        x = observed['x']

        l_x_zy = (self.l_x_zy.construct(
            z=tf.reshape(z, (-1, self.n_z)),
            y=tf.reshape(tf.tile(y, [1, tf.shape(z)[1], 1]),
                         (-1, self.n_y)))).tensor
        l_x_zy = tf.reshape(l_x_zy, [-1, tf.shape(z)[1], self.n_x])
        log_px_zy = tf.reduce_sum(
            bernoulli.logpdf(tf.expand_dims(x, 1), l_x_zy, eps=1e-6), 2)
        log_pz = tf.reduce_sum(norm.logpdf(z), 2)
        log_py = tf.log(tf.constant(1., tf.float32) / self.n_y)
        return log_px_zy + log_pz + log_py


class M2Labeled(M2):
    """
    The M2 model with labels `y` as observed variables.
    """
    def __init__(self, n_x, n_y, n_z):
        super(M2Labeled, self).__init__(n_x, n_y, n_z)

    def log_prob(self, latent, observed):
        # z: (batch_size, n_samples, n_z)
        z = latent['z']
        # y: (batch_size, n_y), x: (batch_size, n_x)
        y, x = observed['y'], observed['x']

        y_in = tf.reshape(
            tf.tile(tf.expand_dims(y, 1), [1, tf.shape(z)[1], 1]),
            (-1, self.n_y))
        l_x_zy = (self.l_x_zy.construct(z=tf.reshape(z, (-1, self.n_z)),
                                        y=y_in)).tensor
        l_x_zy = tf.reshape(l_x_zy, [-1, tf.shape(z)[1], self.n_x])
        log_px_zy = tf.reduce_sum(
            bernoulli.logpdf(tf.expand_dims(x, 1), l_x_zy, eps=1e-6), 2)
        log_pz = tf.reduce_sum(norm.logpdf(z), 2)
        log_py = tf.log(tf.constant(1., tf.float32) / self.n_y)
        return log_px_zy + log_pz + log_py


def q_net_labeled(qz_xy, qz_mean, qz_logstd, n_x, n_y, n_z, n_samples):
    lx = InputLayer((None, n_x))
    ly = InputLayer((None, n_y))
    lz_xy = PrettyTensor({'x': lx, 'y': ly}, qz_xy)
    lz_mean_2d = PrettyTensor({'z': lz_xy}, qz_mean)
    lz_logstd_2d = PrettyTensor({'z': lz_xy}, qz_logstd)
    lz_mean = PrettyTensor({'z_mean': lz_mean_2d}, pt.template('z_mean').
                           reshape((-1, 1, n_z)))
    lz_logstd = PrettyTensor({'z_logstd': lz_logstd_2d},
                             pt.template('z_logstd').
                             reshape((-1, 1, n_z)))
    lz = ReparameterizedNormal([lz_mean, lz_logstd], n_samples=n_samples)
    return lx, ly, lz


def q_net_unlabeled(qy_x, qz_xy, qz_mean, qz_logstd, n_x, n_y, n_z, n_samples):
    lx = InputLayer((None, n_x))
    ly_x_2d = PrettyTensor({'x': lx}, qy_x)
    ly_x = PrettyTensor({'y': ly_x_2d}, pt.template('y').
                        reshape((-1, 1, n_y)))
    ly = Discrete(ly_x, n_classes=n_y)
    ly_2d = PrettyTensor({'y': ly}, pt.template('y').reshape((-1, n_y)))
    lz_xy = PrettyTensor({'x': lx, 'y': ly_2d}, qz_xy)
    lz_mean_2d = PrettyTensor({'z': lz_xy}, qz_mean)
    lz_logstd_2d = PrettyTensor({'z': lz_xy}, qz_logstd)
    lz_mean = PrettyTensor({'z_mean': lz_mean_2d}, pt.template('z_mean').
                           reshape((-1, 1, n_z)))
    lz_logstd = PrettyTensor({'z_logstd': lz_logstd_2d},
                             pt.template('z_logstd').
                             reshape((-1, 1, n_z)))
    lz = ReparameterizedNormal([lz_mean, lz_logstd], n_samples)
    return lx, ly, lz


def q_net(n_x, n_y, n_z, n_samples):
    """
    Build the recognition network q(y, z|x) = q(y|x)q(z|x, y) used as
    variational posterior.

    :param n_x: Int. The dimension of observed variable x.
    :param n_y: Int. The dimension of latent variable y, i.e., the number of
        classes.
    :param n_z: Int. The dimension of latent variable z.
    :param n_samples: Int. Number of samples of latent variable z used.

    :return: All :class:`Layer` instances needed.
    """
    with pt.defaults_scope(activation_fn=tf.nn.relu):
        qy_x = (pt.template('x').
                fully_connected(500).
                fully_connected(500).
                fully_connected(n_y, activation_fn=tf.nn.softmax))

        qz_xy = (pt.template('x').
                 join([pt.template('y')]).
                 fully_connected(500).
                 fully_connected(500))

        qz_mean = pt.template('z').fully_connected(n_z, activation_fn=None)
        qz_logstd = pt.template('z').fully_connected(n_z, activation_fn=None)

    lx_labeled, ly_labeled, lz_labeled = q_net_labeled(
        qz_xy, qz_mean, qz_logstd, n_x, n_y, n_z, n_samples)
    labeled_observed = {'x': lx_labeled, 'y': ly_labeled}
    labeled_latent = {'z': lz_labeled}

    lx_unlabeled, ly_unlabeled, lz_unlabeled = q_net_unlabeled(
        qy_x, qz_xy, qz_mean, qz_logstd, n_x, n_y, n_z, n_samples)
    unlabeled_observed = {'x': lx_unlabeled}
    unlabeled_latent = {'y': ly_unlabeled, 'z': lz_unlabeled}

    return labeled_observed, labeled_latent, unlabeled_observed, \
        unlabeled_latent, qy_x


if __name__ == "__main__":
    tf.set_random_seed(1234)

    # Load MNIST
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'data', 'mnist.pkl.gz')
    np.random.seed(1234)
    x_labeled, t_labeled, x_unlabeled, x_test, t_test = \
        dataset.load_mnist_semi_supervised(data_path, one_hot=True)
    x_test = np.random.binomial(1, x_test, size=x_test.shape).astype('float32')
    n_x = x_labeled.shape[1]
    n_y = t_labeled.shape[1]

    # Define model parameters
    n_z = 100

    # Define training/evaluation parameters
    lb_samples = 1
    ll_samples = 1000
    beta = 1200.
    epoches = 3000
    batch_size = 100
    test_batch_size = 100
    iters = x_unlabeled.shape[0] // batch_size
    test_iters = x_test.shape[0] // test_batch_size
    test_freq = 10
    learning_rate = 0.0003
    anneal_lr_freq = 200
    anneal_lr_rate = 0.75

    # Build the computation graph
    learning_rate_ph = tf.placeholder(tf.float32, shape=[])
    n_samples = tf.placeholder(tf.int32, shape=[])
    optimizer = tf.train.AdamOptimizer(learning_rate_ph, epsilon=1e-4)
    with tf.variable_scope("model") as scope:
        m2_labeled = M2Labeled(n_x, n_y, n_z)
    with tf.variable_scope("model", reuse=True) as scope:
        m2_unlabeled = M2Unlabeled(n_x, n_y, n_z)
    labeled_observed, labeled_latent, unlabeled_observed, unlabeled_latent, \
        qy_x = q_net(n_x, n_y, n_z, n_samples)

    x_labeled_ph = tf.placeholder(tf.float32, shape=(None, n_x))
    y_labeled_ph = tf.placeholder(tf.float32, shape=(None, n_y))
    labeled_inputs = {'x': x_labeled_ph, 'y': y_labeled_ph}
    labeled_grads, labeled_lower_bound = advi(
        m2_labeled, labeled_inputs, labeled_observed,
        labeled_latent, optimizer)
    labeled_infer = optimizer.apply_gradients(labeled_grads)

    x_unlabeled_ph = tf.placeholder(tf.float32, shape=(None, n_x))
    unlabeled_inputs = {'x': x_unlabeled_ph}
    unlabeled_grads, unlabeled_lower_bound = advi(
        m2_unlabeled, unlabeled_inputs, unlabeled_observed,
        unlabeled_latent, optimizer)
    unlabeled_infer = optimizer.apply_gradients(unlabeled_grads)

    # Build classifier
    y = qy_x.construct(x=x_labeled_ph).tensor
    pred_y = tf.argmax(y, 1)
    acc = tf.reduce_sum(
        tf.cast(tf.equal(pred_y, tf.argmax(y_labeled_ph, 1)), tf.float32) /
        tf.cast(tf.shape(y_labeled_ph)[0], tf.float32))
    log_qy_x = discrete.logpdf(y_labeled_ph, y, eps=1e-8)
    classifier_cost = -beta * tf.reduce_mean(log_qy_x)
    classifier_grads = optimizer.compute_gradients(classifier_cost)
    classifier_updates = optimizer.apply_gradients(classifier_grads)

    params = tf.trainable_variables()
    for i in params:
        print(i.name, i.get_shape())

    init = tf.initialize_all_variables()

    # Run the inference
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(1, epoches + 1):
            time_epoch = -time.time()
            if epoch % anneal_lr_freq == 0:
                learning_rate *= anneal_lr_rate
            # np.random.shuffle(x_unlabeled)
            lbs_labeled = []
            lbs_unlabeled = []
            train_accs = []
            train_ccs = []
            for t in range(iters):
                x_labeled_batch = np.random.binomial(
                    n=1, p=x_labeled, size=x_labeled.shape).astype('float32')
                y_labeled_batch = t_labeled
                x_unlabeled_batch = x_unlabeled[t * batch_size:
                                                (t + 1) * batch_size]
                x_unlabeled_batch = np.random.binomial(
                    n=1, p=x_unlabeled_batch,
                    size=x_unlabeled_batch.shape).astype('float32')
                _, lb_labeled, _, cc, _, lb_unlabeled, train_acc = sess.run(
                    [labeled_infer, labeled_lower_bound,
                     classifier_updates, classifier_cost,
                     unlabeled_infer, unlabeled_lower_bound, acc],
                    feed_dict={x_labeled_ph: x_labeled_batch,
                               y_labeled_ph: y_labeled_batch,
                               x_unlabeled_ph: x_unlabeled_batch,
                               learning_rate_ph: learning_rate,
                               n_samples: lb_samples})
                lbs_labeled.append(lb_labeled)
                lbs_unlabeled.append(lb_unlabeled)
                train_accs.append(train_acc)
                train_ccs.append(cc)
            time_epoch += time.time()
            print('Epoch {} ({:.1f}s), Lower bound: labeled = {}, '
                  'unlabeled = {} Accuracy: {:.2f}%, cc = {}'.
                  format(epoch, time_epoch, np.mean(lbs_labeled),
                         np.mean(lbs_unlabeled), np.mean(train_accs) * 100.,
                         np.mean(train_ccs)))
            if epoch % test_freq == 0:
                time_test = -time.time()
                test_lls_labeled = []
                test_lls_unlabeled = []
                test_accs = []
                for t in range(test_iters):
                    test_x_batch = x_test[
                        t * test_batch_size: (t + 1) * test_batch_size]
                    test_y_batch = t_test[
                        t * test_batch_size: (t + 1) * test_batch_size]
                    test_ll_labeled, test_ll_unlabeled, test_acc = sess.run(
                        [labeled_lower_bound, unlabeled_lower_bound, acc],
                        feed_dict={x_labeled_ph: test_x_batch,
                                   y_labeled_ph: test_y_batch,
                                   x_unlabeled_ph: test_x_batch,
                                   n_samples: lb_samples})
                    test_lls_labeled.append(test_ll_labeled)
                    test_lls_unlabeled.append(test_ll_unlabeled)
                    test_accs.append(test_acc)
                time_test += time.time()
                print('>>> TEST ({:.1f}s)'.format(time_test))
                print('>> Test lower bound: labeled = {}, unlabeled = {}'.
                      format(np.mean(test_lls_labeled),
                             np.mean(test_lls_unlabeled)))
                print('>> Test accuracy: {:.2f}%'.format(
                    100. * np.mean(test_accs)))
