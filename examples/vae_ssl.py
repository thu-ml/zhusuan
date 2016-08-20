#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import sys
import os
import itertools

import tensorflow as tf
import prettytensor as pt
import six
from six.moves import range, zip, map
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from zhusuan.distributions import norm, bernoulli
    from zhusuan.utils import log_mean_exp, ensure_dim_match
    from zhusuan.layers import *
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
        with pt.defaults_scope(activation_fn=tf.nn.relu,
                               scale_after_normalization=True):
            l_x_zy = (pt.template('z').join([pt.template('y')]).
                      fully_connected(500).
                      # batch_normalize().
                      fully_connected(500).
                      # batch_normalize().
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

        # shape: (batch_size * n_samples, n_x)
        l_x_zy = (self.l_x_zy.construct(
            z=tf.reshape(z, (-1, self.n_z)),
            y=tf.reshape(
                tf.tile(y, tf.pack([1, tf.shape(z)[1], 1])),
                (-1, self.n_y)
            ))).tensor
        # shape: (batch_size, n_samples, n_x)
        l_x_zy = tf.reshape(l_x_zy, tf.pack([-1, tf.shape(z)[1], self.n_x]))
        # shape: (batch_size, n_samples)
        log_px_zy = tf.reduce_sum(
            bernoulli.logpdf(tf.expand_dims(x, 1), l_x_zy, eps=1e-6), 2)
        # shape: (batch_size, n_samples)
        log_pz = tf.reduce_sum(norm.logpdf(z), 2)
        # shape: (1,)
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

        # shape: (batch_size * n_samples, n_y)
        y_in = tf.reshape(
            tf.tile(tf.expand_dims(y, 1), tf.pack([1, tf.shape(z)[1], 1])),
            (-1, self.n_y))
        # shape: (batch_size * n_samples, n_x)
        l_x_zy = (self.l_x_zy.construct(z=tf.reshape(z, (-1, self.n_z)),
                                        y=y_in)).tensor
        # shape: (batch_size, n_samples, n_x)
        l_x_zy = tf.reshape(l_x_zy, tf.pack([-1, tf.shape(z)[1], self.n_x]))
        # shape: (batch_size, n_samples)
        log_px_zy = tf.reduce_sum(
            bernoulli.logpdf(tf.expand_dims(x, 1), l_x_zy, eps=1e-6), 2)
        # shape: (batch_size, n_samples)
        log_pz = tf.reduce_sum(norm.logpdf(z), 2)
        # shape: (1,)
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

    # lx_labeled: (batch_size, n_x)
    # ly_labeled: (batch_size, n_y)
    # lz_labeled: (batch_size, n_samples, n_z)
    lx_labeled, ly_labeled, lz_labeled = q_net_labeled(
        qz_xy, qz_mean, qz_logstd, n_x, n_y, n_z, n_samples)
    labeled_observed = {'x': lx_labeled, 'y': ly_labeled}
    labeled_latent = {'z': lz_labeled}

    # lx_unlabeled: (batch_size, n_x)
    # ly_unlabeled: (batch_size, n_samples, n_y)
    # lz_unlabeled: (batch_size, n_samples, n_z)
    lx_unlabeled, ly_unlabeled, lz_unlabeled = q_net_unlabeled(
        qy_x, qz_xy, qz_mean, qz_logstd, n_x, n_y, n_z, n_samples)
    unlabeled_observed = {'x': lx_unlabeled}
    unlabeled_latent = {'y': ly_unlabeled, 'z': lz_unlabeled}

    return labeled_observed, labeled_latent, unlabeled_observed, \
        unlabeled_latent


def advi(model, observed_inputs, observed_layers, latent_layers,
         optimizer=tf.train.AdamOptimizer()):
    """
    Implements the automatic differentiation variational inference (ADVI)
    algorithm. For now we assume all latent variables have been transformed in
    the model definition to have their support on R^n.

    :param model: A model object that has a method logprob(latent, observed)
        to compute the log joint likelihood of the model.
    :param observed_inputs: A dictionary. Given inputs to the observed layers.
    :param observed_layers: A dictionary. The observed layers.
    :param latent_layers: A dictionary. The latent layers.
    :param optimizer: Tensorflow optimizer object. Default to be
        AdamOptimizer.

    :return: Tensorflow gradients that can be applied using
        `tf.train.Optimizer.apply_gradients`
    :return: A 0-D Tensor. The variational lower bound.
    """
    if list(six.iterkeys(observed_inputs)) != list(
            six.iterkeys(observed_layers)):
        raise ValueError("Observed layers and inputs don't match.")

    # add all observed (variable, input) pairs into inputs
    inputs = {}
    for name, layer in six.iteritems(observed_layers):
        inputs[layer] = observed_inputs[name]

    # get discrete latent layers
    latent_k, latent_v = map(list, zip(*six.iteritems(latent_layers)))
    discrete_latent_layers = dict(filter(lambda x: isinstance(x[1], Discrete),
                                         six.iteritems(latent_layers)))

    if discrete_latent_layers:
        # Discrete latent layers exists
        discrete_latent_k, discrete_latent_v = map(list, zip(
            *six.iteritems(discrete_latent_layers)))

        # get all configurations of discrete latent variables
        all_disc_latent_configs = []
        for layer in discrete_latent_v:
            if layer.n_samples > 1:
                raise ValueError("advi() doesn't support Discrete latent "
                                 "layers with n_samples (=%d) > 1" %
                                 layer.n_samples)
            tmp = []
            for i in range(layer.n_classes):
                layer_input = tf.expand_dims(tf.expand_dims(tf.one_hot(
                    i, depth=layer.n_classes, dtype=tf.float32), 0), 0)
                tmp.append(layer_input)
            all_disc_latent_configs.append(tmp)
        # cartesian products
        all_disc_latent_configs = itertools.product(*all_disc_latent_configs)

        # feed all configurations of inputs
        weighted_lbs = []
        for discrete_latent_inputs in all_disc_latent_configs:
            _inputs = inputs.copy()
            discrete_latent_inputs = dict(zip(discrete_latent_v,
                                              discrete_latent_inputs))
            _inputs.update(discrete_latent_inputs)
            # ensure the batch_size dimension matches
            _inputs_k, _inputs_v = zip(*six.iteritems(_inputs))
            _inputs_v = ensure_dim_match(_inputs_v, 0)
            _inputs = dict(zip(_inputs_k, _inputs_v))
            # size: continuous layers (batch_size, n_samples, n_dim)
            #       discrete layers (batch_size, 1, n_dim)
            outputs = get_output(latent_v, _inputs)
            latent_outputs = dict(zip(latent_k, map(lambda x: x[0], outputs)))
            # size: continuous layer (batch_size, n_samples)
            #       discrete layers (batch_size, 1)
            latent_logpdfs = dict(zip(latent_k, map(lambda x: x[1], outputs)))
            # size: (batch_size, n_samples)
            lower_bound = model.log_prob(latent_outputs, observed_inputs) - \
                sum(six.itervalues(latent_logpdfs))
            discrete_latent_logpdfs = [latent_logpdfs[i]
                                       for i in discrete_latent_k]
            w = tf.exp(sum(discrete_latent_logpdfs))
            weighted_lbs.append(lower_bound * w)
        # size: (batch_size, n_samples)
        lower_bound = sum(weighted_lbs)
    else:
        # no Discrete latent layers
        outputs = get_output(latent_v, inputs)
        latent_outputs = dict(zip(latent_k, map(lambda x: x[0], outputs)))
        latent_logpdfs = map(lambda x: x[1], outputs)
        lower_bound = model.log_prob(latent_outputs, observed_inputs) - \
            sum(latent_logpdfs)

    lower_bound = tf.reduce_mean(lower_bound)
    return optimizer.compute_gradients(-lower_bound), lower_bound


if __name__ == "__main__":
    tf.set_random_seed(1234)

    # Load MNIST
    n_labels = 100
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'data', 'mnist.pkl.gz')
    x_train, t_train, x_valid, t_valid, x_test, t_test = \
        dataset.load_mnist_realval(data_path)
    x_train = np.vstack([x_train, x_valid]).astype('float32')
    t_train = np.vstack([t_train, t_valid]).astype('float32')
    x_labeled, t_labeled = x_train[:n_labels], t_train[:n_labels]
    x_unlabeled = x_train[n_labels:]
    np.random.seed(1234)
    x_test = np.random.binomial(1, x_test, size=x_test.shape).astype('float32')
    n_x = x_train.shape[1]
    n_y = t_train.shape[1]

    # Define model parameters
    n_z = 40

    # Define training/evaluation parameters
    lb_samples = 1
    ll_samples = 5000
    beta = 0.01
    epoches = 3000
    batch_size = 100
    test_batch_size = 100
    iters = x_unlabeled.shape[0] // batch_size

    # Build the training computation graph
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001, epsilon=1e-4)
    with tf.variable_scope("model") as scope:
        m2_labeled = M2Labeled(n_x, n_y, n_z)
    with tf.variable_scope("model", reuse=True) as scope:
        m2_unlabeled = M2Unlabeled(n_x, n_y, n_z)
    labeled_observed, labeled_latent, unlabeled_observed, unlabeled_latent = \
        q_net(n_x, n_y, n_z, lb_samples)

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

    params = tf.trainable_variables()
    for i in params:
        print(i.name, i.get_shape())

    init = tf.initialize_all_variables()
    check_op = tf.add_check_numerics_ops()

    # Run the inference
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(1, epoches + 1):
            np.random.shuffle(x_unlabeled)
            lbs_labeled = []
            lbs_unlabeled = [0]
            for t in range(iters):
                x_labeled_batch = np.random.binomial(
                    n=1, p=x_labeled, size=x_labeled.shape).astype('float32')
                y_labeled_batch = t_labeled
                x_unlabeled_batch = x_unlabeled[t * batch_size:
                                                (t + 1) * batch_size]
                x_unlabeled_batch = np.random.binomial(
                    n=1, p=x_unlabeled_batch,
                    size=x_unlabeled_batch.shape).astype('float32')
                _, lb_labeled = sess.run(
                    [labeled_infer, labeled_lower_bound],
                    feed_dict={x_labeled_ph: x_labeled_batch,
                               y_labeled_ph: y_labeled_batch})
                _, lb_unlabeled = sess.run(
                    [unlabeled_infer, unlabeled_lower_bound],
                    feed_dict={x_unlabeled_ph: x_unlabeled_batch})
                # print('Epoch {}, Iter {}, Lower bound: labeled = {}, '
                #       'unlabeled = {}'.format(epoch, t, lb_labeled,
                #                               lb_unlabeled))
                lbs_labeled.append(lb_labeled)
                lbs_unlabeled.append(lb_unlabeled)
            print('Epoch {}, Lower bound: labeled = {}, unlabeled = {}'.format(
                epoch, np.mean(lbs_labeled), np.mean(lbs_unlabeled)))
