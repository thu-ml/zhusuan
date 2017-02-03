#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import sys
import os
import time

import tensorflow as tf
from tensorflow.contrib import layers
from six.moves import range
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import zhusuan as zs
except:
    raise ImportError()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import dataset
except:
    raise ImportError()


@zs.reuse('model')
def M2(observed, n, n_x, n_y, n_z, n_particles, is_training):
    normalizer_params = {'is_training': is_training,
                         'updates_collections': None}
    with StochasticGraph() as model:
        z_mean = tf.zeros([n_particles, n_z])
        z_logstd = tf.zeros([n_particles, n_z])
        z = zs.Normal('z', z_mean, z_logstd, sample_dim=1, n_samples=n)
        y_logits = tf.zeros([n_particles, n_y])
        y = zs.Discrete('y', y_logits, sample_dim=1, n_samples=n)
        lx_zy = layers.fully_connected(
            tf.concat_v2([z, y], 2), 500, normalizer_fn=layers.batch_norm,
            normalizer_params=normalizer_params)
        lx_zy = layers.fully_connected(
            lx_zy, 500, normalizer_fn=layers.batch_norm,
            normalizer_params=normalizer_params)
        x_logits = layers.fully_connected(lx_zy, n_x, activation_fn=None)
        x = zs.Bernoulli('x', x_logits)
    return model


@zs.reuse('variational')
def q_net(observed, n_x, n_y, n_z, n_particles, is_training):
    pass


def log_prob(self, latent, observed, given):
    """
    The log joint probability function.

    :param latent: A dictionary of pairs: (string, Tensor). Each of the
        Tensor has shape (batch_size, n_samples, n_latent).
    :param observed: A dictionary of pairs: (string, Tensor). Each of the
        Tensor has shape (batch_size, n_observed).
    :return: A Tensor of shape (batch_size, n_samples). The joint log
        likelihoods.
    """
    # z: (n_samples, batch_size, n_z)
    z = latent['z']
    x = observed['x']
    # y: (batch_size, n_y), x: (batch_size, n_x)
    if 'y' in observed:
        y = tf.tile(tf.expand_dims(observed['y'], 0),
                    [self.n_particles, 1, 1])
    else:
        y = latent['y']
    x = tf.tile(tf.expand_dims(x, 0), [self.n_particles, 1, 1])
    z_out, x_out = self.model.get_output(
        [self.z, self.x], inputs={self.z: z, self.x: x, self.y: y})
    log_px_zy = tf.reduce_sum(x_out[1], -1)
    log_pz = tf.reduce_sum(z_out[1], -1)
    log_py = tf.log(tf.constant(1., tf.float32) / tf.cast(tf.shape(y)[-1],
                                                          tf.float32))
    return log_px_zy + log_pz + log_py


def q_net(n_x, n_y, n_z, n_particles, is_training):
    """
    Build the recognition network q(y, z|x) = q(y|x)q(z|x, y) used as
    variational posterior.

    :param n_z: Int. The dimension of latent variable z.
    :param n_particles: Tensor or int. Number of samples of latent variables.
    :param is_training: Bool.
    :return: All :class:`Layer` instances needed.
    """
    normalizer_params = {'is_training': is_training,
                         'updates_collections': None}
    with StochasticGraph() as variational:
        l_x = tf.placeholder(tf.float32, shape=(None, n_x))
        l_y = tf.placeholder(tf.float32, shape=(None, n_y))
        lz_xy = layers.fully_connected(tf.concat_v2([l_x, l_y], 1), 500,
                                       scope='lz_xy1')
        lz_xy = layers.fully_connected(lz_xy, 500,
                                       scope='lz_xy2')
        lz_mean = layers.fully_connected(lz_xy, n_z, activation_fn=None,
                                         scope='lz_mean')
        lz_logstd = layers.fully_connected(lz_xy, n_z, activation_fn=None,
                                           scope='lz_logstd')
        lz = Normal(lz_mean, lz_logstd, sample_dim=0, n_samples=n_particles,
                    reparameterized=False)

        lx_u = tf.placeholder(tf.float32, shape=(None, n_x))
        qy_x = layers.fully_connected(
            lx_u, 500,
            normalizer_fn=layers.batch_norm,
            normalizer_params=normalizer_params
        )
        qy_x = layers.fully_connected(
            qy_x, 500,
            normalizer_fn=layers.batch_norm,
            normalizer_params=normalizer_params
        )
        qy_x = layers.fully_connected(qy_x, n_y, activation_fn=None)
        ly_u = Discrete(qy_x, sample_dim=0, n_samples=n_particles)
        lz_xy_u = layers.fully_connected(
            tf.concat_v2([tf.tile(tf.expand_dims(lx_u, 0),
                                  [n_particles, 1, 1]),
                          ly_u.value], 2),
            500, reuse=True, scope='lz_xy1')
        lz_mean_u = layers.fully_connected(lz_xy_u, n_z, activation_fn=None,
                                           reuse=True, scope='lz_mean')
        lz_logstd_u = layers.fully_connected(lz_xy_u, n_z, activation_fn=None,
                                             reuse=True, scope='lz_logstd')
        lz_u = Normal(lz_mean_u, lz_logstd_u, reparameterized=False)

    return variational, l_x, l_y, lz, lx_u, ly_u, lz_u, qy_x


if __name__ == "__main__":
    tf.set_random_seed(1234)

    # Load MNIST
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'data', 'mnist.pkl.gz')
    np.random.seed(1234)
    x_labeled, t_labeled, x_unlabeled, x_test, t_test = \
        dataset.load_mnist_semi_supervised(data_path, one_hot=True)
    x_test = np.random.binomial(1, x_test, size=x_test.shape).astype('float32')
    n_labeled, n_x = x_labeled.shape
    n_y = 10

    # Define model parameters
    n_z = 100

    # Define training/evaluation parameters
    ll_samples = 10
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
    is_training = tf.placeholder(tf.bool, shape=[], name='is_training')
    learning_rate_ph = tf.placeholder(tf.float32, shape=[], name='lr')
    n_particles = tf.placeholder(tf.int32, shape=[], name='n_particles')
    optimizer = tf.train.AdamOptimizer(learning_rate_ph)

    # Labeled
    x_labeled_ph = tf.placeholder(tf.float32, shape=(None, n_x), name='x_l')
    y_labeled_ph = tf.placeholder(tf.float32, shape=(None, n_y), name='y_l')

    n = tf.shape(x_labeled_ph)[0]
    with tf.variable_scope('model') as scope:
        m2 = M2(n_x, n_y, n_z, n, n_particles, is_training)

    variational, lx, ly, lz, lx_u, ly_u, lz_u, qy_x = \
        q_net(n_x, n_y, n_z, n_particles, is_training)
    inputs = {lx: x_labeled_ph, ly: y_labeled_ph, lx_u: x_labeled_ph}
    z_outputs, y_outputs = variational.get_output([lz, qy_x], inputs)

    labeled_latent = {'z': [z_outputs[0], tf.reduce_sum(z_outputs[1], -1)]}
    labeled_observed = {'x': x_labeled_ph, 'y': y_labeled_ph}
    labeled_cost, labeled_log_likelihood = rws(
        m2, labeled_observed, labeled_latent, reduction_indices=0)
    labeled_cost = tf.reduce_mean(labeled_cost)
    labeled_log_likelihood = tf.reduce_mean(labeled_log_likelihood)

    # Unlabeled
    x_unlabeled_ph = tf.placeholder(tf.float32, shape=(None, n_x), name='x_u')
    inputs = {lx_u: x_unlabeled_ph}
    y_u_outputs, z_u_outputs = variational.get_output([ly_u, lz_u], inputs)

    unlabeled_latent = {'z': [z_u_outputs[0],
                              tf.reduce_sum(z_u_outputs[1], -1)],
                        'y': [y_u_outputs[0], y_u_outputs[1]]}
    unlabeled_observed = {'x': x_unlabeled_ph}

    unlabeled_cost, unlabeled_log_likelihood = rws(
        m2, unlabeled_observed, unlabeled_latent, reduction_indices=0)
    unlabeled_cost = tf.reduce_mean(unlabeled_cost)
    unlabeled_log_likelihood = tf.reduce_mean(unlabeled_log_likelihood)

    # Build classifier
    pred_y = tf.argmax(y_outputs[0], 1)
    acc = tf.reduce_sum(
        tf.cast(tf.equal(pred_y, tf.argmax(y_labeled_ph, 1)), tf.float32) /
        tf.cast(tf.shape(y_labeled_ph)[0], tf.float32))
    log_qy_x = discrete.logpmf(y_labeled_ph, y_outputs[0])
    classifier_cost = -beta * tf.reduce_mean(log_qy_x)

    # Gather gradients
    cost = (labeled_cost + unlabeled_cost + classifier_cost) / 2.
    grads = optimizer.compute_gradients(cost)
    infer = optimizer.apply_gradients(grads)

    params = tf.trainable_variables()
    for i in params:
        print(i.name, i.get_shape())

    init = tf.global_variables_initializer()
    # graph_writer = tf.train.SummaryWriter('/home/ishijiaxin/log',
    #                                       tf.get_default_graph())

    # Run the inference
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(1, epoches + 1):
            time_epoch = -time.time()
            if epoch % anneal_lr_freq == 0:
                learning_rate *= anneal_lr_rate
            np.random.shuffle(x_unlabeled)
            lbs_labeled = []
            lbs_unlabeled = []
            train_accs = []
            for t in range(iters):
                labeled_indices = np.random.randint(0, n_labeled,
                                                    size=batch_size)
                x_labeled_batch = x_labeled[labeled_indices]
                y_labeled_batch = t_labeled[labeled_indices]
                x_unlabeled_batch = x_unlabeled[t * batch_size:
                                                (t + 1) * batch_size]
                x_labeled_batch = np.random.binomial(
                    n=1, p=x_labeled_batch,
                    size=x_labeled_batch.shape).astype('float32')
                x_unlabeled_batch = np.random.binomial(
                    n=1, p=x_unlabeled_batch,
                    size=x_unlabeled_batch.shape).astype('float32')
                _, lb_labeled, lb_unlabeled, train_acc = sess.run(
                    [infer, labeled_log_likelihood, unlabeled_log_likelihood,
                     acc],
                    feed_dict={x_labeled_ph: x_labeled_batch,
                               y_labeled_ph: y_labeled_batch,
                               x_unlabeled_ph: x_unlabeled_batch,
                               learning_rate_ph: learning_rate,
                               n_particles: ll_samples,
                               is_training: True})
                lbs_labeled.append(lb_labeled)
                lbs_unlabeled.append(lb_unlabeled)
                train_accs.append(train_acc)
            time_epoch += time.time()
            print('Epoch {} ({:.1f}s), Lower bound: labeled = {}, '
                  'unlabeled = {} Accuracy: {:.2f}%'.
                  format(epoch, time_epoch, np.mean(lbs_labeled),
                         np.mean(lbs_unlabeled), np.mean(train_accs) * 100.))
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
                        [labeled_log_likelihood, unlabeled_log_likelihood,
                         acc],
                        feed_dict={x_labeled_ph: test_x_batch,
                                   y_labeled_ph: test_y_batch,
                                   x_unlabeled_ph: test_x_batch,
                                   n_particles: ll_samples,
                                   is_training: False})
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
