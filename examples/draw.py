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
from scipy.misc import imsave
from six.moves import range, map
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from zhusuan.distributions import norm, bernoulli
    from zhusuan.layers import *
    from zhusuan.variational import advi
    from zhusuan.evaluation import is_loglikelihood
except:
    raise ImportError()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import dataset
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

    def log_prob(self, latent, observed, given):
        """
        The log joint probability function.

        :param latent: A dictionary of pairs: (string, Tensor). Each of the
            Tensor has shape (batch_size, n_samples, n_latent).
        :param observed: A dictionary of pairs: (string, Tensor). Each of the
            Tensor has shape (batch_size, n_observed).
        :param given:

        :return: A Tensor of shape (batch_size, n_samples). The joint log
            likelihoods.
        """
        zs = six.itervalues(latent)
        x = observed['x']
        c = given['c']

        log_px_z = tf.reduce_sum(
            bernoulli.logpdf(tf.expand_dims(x, 1), tf.sigmoid(c), eps=1e-6), 2)
        log_pz = sum(tf.reduce_sum(norm.logpdf(z), 2) for z in zs)
        return log_px_z + log_pz


def q_net(n_x, n_z, n_samples, n_steps):
    lx = InputLayer((None, n_x), name='lx')
    dec_h_t = InputLayer((None, 256), tf.zeros((100, 256)), name='decht')
    enc_h_t = InputLayer((None, 256), tf.zeros((100, 256)), name='encht')
    enc_s_t = ListLayer([enc_h_t,
                         InputLayer((None, 256), tf.zeros((100, 256)), name='encst')])
    dec_s_t = ListLayer([dec_h_t,
                         InputLayer((None, 256), tf.zeros((100, 256)), name='decst')])
    c_t = InputLayer((None, n_x), tf.zeros((100, n_x)), name='ct')
    enc_lstm = (pt.template('read').join([pt.template('dec_h')]).lstm_cell((pt.template('c'), pt.template('h')), 256))
    dec_lstm = (pt.template('z').lstm_cell((pt.template('c'), pt.template('h')), 256))
    lz_list = []
    read_attn_nets = [pt.template('dec_h').fully_connected(1, activation_fn=None) for j in range(5)]
    write_attn_nets = [pt.template('dec_h').fully_connected(1, activation_fn=None) for j in range(5)]
    mean_net = pt.template('z').fully_connected(n_z, activation_fn=None).reshape((-1, 1, n_z))
    logstd_net = pt.template('z').fully_connected(n_z, activation_fn=None).reshape((-1, 1, n_z))
    write_patch_net = pt.template('dec_h').fully_connected(5*5, activation_fn=None)
    for i in range(n_steps):
        read_attn = list(map(lambda r: PrettyTensor({'dec_h': dec_h_t}, r), read_attn_nets))
        # shape: (batch_size, 2*read_n*read_n)
        read = ReadAttentionLayer([lx, c_t] + read_attn, width=28, height=28, read_n=5)
        enc_hs_t = PrettyTensor({'read': read, 'dec_h': dec_h_t,
                                 'c': ListIndexLayer(enc_s_t, 0, name='c'),
                                 'h': ListIndexLayer(enc_s_t, 1, name='h')},
                                enc_lstm)
        # shape: (batch_size, 256)
        enc_h_t = ListIndexLayer(enc_hs_t, 0, name='ht')
        # type: tuple, (c, h)
        enc_s_t = ListIndexLayer(enc_hs_t, 1, name='st')
        lz_mean = PrettyTensor({'z': enc_h_t}, mean_net)
        lz_logstd = PrettyTensor({'z': enc_h_t}, logstd_net)
        lz = ReparameterizedNormal([lz_mean, lz_logstd], n_samples)
        lz_2d = PrettyTensor({'lz': lz}, pt.template('lz').reshape([-1, n_z]))
        lz_list.append(lz)
        dec_hs_t = PrettyTensor({'z': lz_2d, 'c': ListIndexLayer(dec_s_t, 0, name='dec_c'),
                                 'h': ListIndexLayer(dec_s_t, 1, name='dec_h')},
                                dec_lstm)
        dec_h_t = ListIndexLayer(dec_hs_t, 0, name='dec_ht')
        dec_s_t = ListIndexLayer(dec_hs_t, 1, name='dec_st')
        # shape: (batch_size, write_n*write_n)
        write_patch = PrettyTensor({'dec_h': dec_h_t}, write_patch_net)
        write_attn = list(map(lambda r: PrettyTensor({'dec_h': dec_h_t}, r), write_attn_nets))
        c_t = WriteAttentionLayer([c_t, write_patch] + write_attn, width=28, height=28, write_n=5)
    c_t = PrettyTensor({'c_t': c_t}, pt.template('c_t').reshape((-1, n_samples, n_x)))
    return lx, lz_list, c_t


if __name__ == "__main__":
    tf.set_random_seed(1237)

    # Load MNIST
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'data', 'mnist.pkl.gz')
    x_train, t_train, x_valid, t_valid, x_test, t_test = \
        dataset.load_mnist_realval(data_path)
    x_train = np.vstack([x_train, x_valid]).astype('float32')
    np.random.seed(1234)
    x_test = np.random.binomial(1, x_test, size=x_test.shape).astype('float32')
    n_x = x_train.shape[1]

    # Define model parameters
    n_z = 40
    n_steps = 32

    # Define training/evaluation parameters
    lb_samples = 1
    ll_samples = 5000
    epoches = 3000
    batch_size = 100
    test_batch_size = 100
    iters = x_train.shape[0] // batch_size
    test_iters = x_test.shape[0] // test_batch_size
    test_freq = 10
    learning_rate = 0.001
    anneal_lr_freq = 200
    anneal_lr_rate = 0.75

    # Build the training computation graph
    learning_rate_ph = tf.placeholder(tf.float32, shape=[])
    x = tf.placeholder(tf.float32, shape=(None, x_train.shape[1]))
    n_samples = tf.placeholder(tf.int32, shape=())
    optimizer = tf.train.AdamOptimizer(learning_rate_ph, epsilon=1e-4)
    with pt.defaults_scope(phase=pt.Phase.train):
        with tf.variable_scope("model") as scope:
            train_model = M1(n_z, x_train.shape[1])
        with tf.variable_scope("variational") as scope:
            lx, lz_list, c_t = q_net(n_x, n_z, n_samples, n_steps)
    lz_dic = dict(('z_' + str(i), lz_list[i]) for i in range(len(lz_list)))
    grads, lower_bound = advi(
        train_model, {'x': x}, {'x': lx}, lz_dic, {'c': c_t}, optimizer)
    infer = optimizer.apply_gradients(grads)

    # Build the evaluation computation graph
    with pt.defaults_scope(phase=pt.Phase.test):
        with tf.variable_scope("model", reuse=True) as scope:
            eval_model = M1(n_z, x_train.shape[1])
        with tf.variable_scope("variational", reuse=True) as scope:
            lx, lz_list, c_t = q_net(n_x, n_z, n_samples, n_steps)
    lz_dic = dict(('z_' + str(i), lz_list[i]) for i in range(len(lz_list)))
    _, eval_lower_bound = advi(
        eval_model, {'x': x}, {'x': lx}, lz_dic, {'c': c_t}, optimizer)
    samples = get_output(c_t, inputs={lx: x})
    samples_gen = get_output(c_t, inputs=dict(
        (lz_list[i], tf.random_normal((batch_size, lb_samples, n_z), mean=0, stddev=1)) for i in range(len(
            lz_list))))
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
            np.random.shuffle(x_train)
            lbs = []
            for t in range(iters):
                x_batch = x_train[t * batch_size:(t + 1) * batch_size]
                x_batch = np.random.binomial(
                    n=1, p=x_batch, size=x_batch.shape).astype('float32')
                _, lb = sess.run([infer, lower_bound],
                                 feed_dict={x: x_batch,
                                            learning_rate_ph: learning_rate,
                                            n_samples: lb_samples})
                lbs.append(lb)
                if t == 1:
                    imgs = sess.run(tf.sigmoid(samples[0]), feed_dict={x: x_batch, n_samples: lb_samples})
                    imgs = imgs.reshape((10, 10, 28, 28)).transpose(
                        1, 2, 0, 3).reshape((10 * 28, 10 * 28))
                    imsave('reconst_%d.png' % epoch, imgs)
                    imgs = sess.run(tf.sigmoid(samples_gen[0]), feed_dict={n_samples: lb_samples})
                    imgs = imgs.reshape((10, 10, 28, 28)).transpose(1, 2, 0, 3).reshape((10 * 28, 10 * 28))
                    imsave('samples_%d.png' % epoch, imgs)
            time_epoch += time.time()
            print('Epoch {} ({:.1f}s): Lower bound = {}'.format(
                epoch, time_epoch, np.mean(lbs)))
            if epoch % test_freq == 0:
                time_test = -time.time()
                test_lbs = []
                for t in range(test_iters):
                    test_x_batch = x_test[
                        t * test_batch_size: (t + 1) * test_batch_size]
                    test_lb = sess.run(eval_lower_bound,
                                       feed_dict={x: test_x_batch,
                                                  n_samples: lb_samples})

                    test_lbs.append(test_lb)
                time_test += time.time()
                print('>>> TEST ({:.1f}s)'.format(time_test))
                print('>> Test lower bound = {}'.format(np.mean(test_lbs)))

