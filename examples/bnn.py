#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import sys
import os
import time
import pdb

import tensorflow as tf
import prettytensor as pt
from six.moves import range
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
    The bayesian neural network model in PBP

    :param n_x: Int. The dimension of observed variables (x).
    """
    def __init__(self, n_x, sample_num):
        self.n_x = n_x
        self.lx = InputLayer((None, n_x))
        self.l1 = BayesianNNLayer(self.lx, 50, sample_num = sample_num)
        self.ly = BayesianNNLayer(self.l1, 1, sample_num = sample_num)
        self.scale = tf.Variable(1., trainable = True, name = 'precision')

    def log_prob(self, x, y):
        """
        The log joint probability function.

        :param X: A dictionary of pairs: (string, Tensor). Each of the
            Tensor has shape (batch_size, n_x).
        :param Y: A dictionary of pairs: (string, Tensor). Each of the
            Tensor has shape (batch_size, n_y).

        :return: A Tensor of shape (batch_size, n_samples). The joint log
            likelihoods.
        """
       
        params = tf.trainable_variables()

        ly = get_output(self.ly, x)
        y1 = tf.tile(tf.expand_dims(y,1), [1, self.l1.sample_num, 1])
        temp = norm.logpdf(y1, tf.squeeze(ly[0], [3]), 1. / self.scale)
        log_px_w = tf.squeeze(temp, [2])

        log_pw = -0.5 * np.log(2 * np.pi) * (len(params)-1)/2.0  \
              - 0.5 * sum([tf.reduce_sum(temp.value()**2) for temp in params if temp.name.startswith(('loc', 'scale'))])
        return log_px_w + log_pw, log_px_w

    def log_prob2(self):
        params = tf.trainable_variables()
        scale = [temp.value() for temp in params if temp.name.startswith('scale')]
        return -0.5 * (np.log(2 * np.pi) + 1) * (len(params)-1)/2.0 \
              - tf.reduce_sum(tf.log(scale))

def get_data(name):
    data = np.loadtxt(name)

    # We obtain the features and the targets
    permutation = np.random.choice(range(data.shape[ 0 ]),
    		data.shape[ 0 ], replace = False)
    size_train = int(np.round(data.shape[ 0 ] * 0.8))
    size_test = int(np.round(data.shape[ 0 ] * 0.9))
    index_train = permutation[ 0 : size_train ] 
    index_test = permutation[ size_train : size_test]
    index_val = permutation[size_test : ]

    X_train, y_train = data[ index_train, : -1], data[ index_train, -1]
    X_val, y_val = data[ index_val, : -1], data[ index_val, -1]
    X_test, y_test = data[ index_test, : -1], data[ index_test, -1]
    
    return X_train, y_train, X_val, y_val, X_test, y_test

if __name__ == "__main__":
    tf.set_random_seed(1237)

    # Load MNIST
    
    x_train, y_train, x_valid, y_valid, x_test, y_test = get_data('data/1boston_housing.txt')
    x_train = np.vstack([x_train, x_valid]).astype('float32')
    y_train = np.vstack([y_train.reshape((len(y_train),1)), y_valid.reshape((len(y_valid),1))]).astype('float32')
    np.random.seed(1234)
    x_test = x_test.astype('float32')
    n_x = x_train.shape[1]


    # Define training/evaluation parameters
    lb_samples = 1
    ll_samples = 100
    epoches = 500
    batch_size = 10
    test_batch_size = 10
    iters = int(np.floor(x_train.shape[0] / float(batch_size)))
    test_iters = int(np.floor(x_test.shape[0] / float(test_batch_size)))
    test_freq = 10
    learning_rate = 0.001
    anneal_lr_freq = 200
    anneal_lr_rate = 0.75

    def build_model(phase, sample_num, reuse=False):
        with pt.defaults_scope(phase=phase):
            with tf.variable_scope("model", reuse=reuse) as scope:
                model = M1(x_train.shape[1], sample_num)
        return model

    # Build the training computation graph
    learning_rate_ph = tf.placeholder(tf.float32, shape=[])
    x = tf.placeholder(tf.float32, shape=(None, x_train.shape[1]))
    y = tf.placeholder(tf.float32, shape=(None, 1))
    n_samples = tf.placeholder(tf.int32, shape=())
    optimizer = tf.train.AdamOptimizer(learning_rate_ph, epsilon=1e-4)
    model = build_model(pt.Phase.train, n_samples)
    lower_bound = tf.reduce_sum(tf.reduce_mean(
        model.log_prob(x,y)[0], 1)) * x_train.shape[0] / batch_size - model.log_prob2()
    grads = optimizer.compute_gradients(-lower_bound)
    infer = optimizer.apply_gradients(grads)

    # Build the evaluation computation graph
    eval_model = build_model(pt.Phase.test, n_samples, reuse=True)
    
    eval_lower_bound = tf.reduce_sum(tf.reduce_mean(
        model.log_prob(x,y)[0], 1)) * x_train.shape[0] / test_batch_size - model.log_prob2()
    eval_log_likelihood = tf.reduce_mean(model.log_prob(x,y)[1])

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
                y_batch = y_train[t * batch_size:(t + 1) * batch_size].reshape((batch_size, 1))
                _, lb = sess.run([infer, lower_bound],
                                 feed_dict={x: x_batch, y:y_batch,
                                            learning_rate_ph: learning_rate,
                                            n_samples: lb_samples})
                lbs.append(lb)
            time_epoch += time.time()
            print('Epoch {} ({:.1f}s): Lower bound = {}'.format(
                epoch, time_epoch, np.mean(lbs)))
            if epoch % test_freq == 0:
                time_test = -time.time()
                test_lbs = []
                test_lls = []
                for t in range(test_iters):
                    test_x_batch = x_test[t * test_batch_size: (t + 1) * test_batch_size]
                    test_y_batch = y_test[t * test_batch_size: (t + 1) * test_batch_size].reshape((test_batch_size, 1))
                    test_lb = sess.run(eval_lower_bound,
                                       feed_dict={x: test_x_batch, y:test_y_batch,
                                                  n_samples: lb_samples})
                    test_ll = sess.run(eval_log_likelihood,
                                       feed_dict={x: test_x_batch, y:test_y_batch,
                                                  n_samples: ll_samples})
                    test_lbs.append(test_lb)
                    test_lls.append(test_ll)
                time_test += time.time()
                print('>>> TEST ({:.1f}s)'.format(time_test))
                print('>> Test lower bound = {}'.format(np.mean(test_lbs)))
                print('>> Test log likelihood = {}'.format(np.mean(test_lls)))
