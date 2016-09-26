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

class Model():
    """
    A simple PBP model
    """
    def __init__(self):
        pass
    def log_prob(self, latent, observed, given):
        """
        The log joint probability function.
        :param latent: A dictionary of pairs: (string, Tensor). Each of the 
            Tensor has shape (1, n_samples, n_latent). The latent variables are global.
        :param observed: A dictionary of pairs: (string, Tensor). Each of 
            the Tenor has shape (batch_size, n_observed).

        :return: A Tensor of shape (batch_size, n_samples). The joint log likelihoods.
        """
        x = observed['x']
        y = observed['y']

        network_output = self.get_output_for(latent, x)
        network_output_1 = tf.squeeze(network_output, [2, 3])
        y_1 = tf.tile(tf.expand_dims(y, 1), [1, tf.shape(network_output_1)[1], 1])
        y_2 = tf.squeeze(y_1, [2])
        log_likelihood = norm.logpdf(y_2, network_output_1, 0.1)
        log_likelihood_1 = tf.reduce_mean(log_likelihood, 0) * 927
        log_likelihood_2 = tf.expand_dims(log_likelihood_1, 0)

        log_prior = sum([tf.reduce_sum(norm.logpdf(item)) for item in latent.values()])
        return log_likelihood_1 + log_prior
 
    def get_output_for(self, latent, x):
        """
        get the network output of x with latent variables.
        :param latent: A dictionary of pairs: (string, Tensor). Each of the 
            Tensor has shape (n_samples, shape_latent). The latent variables are global.
        :param x: A dictionary of pairs: (string, Tensor). Each of the 
            Tensor has shape (batch_size, n_observed_x)
       
        :return: A Tensor of shape (batch_size, n_samples, n_observed_y)
        """
        layer_1 = latent['layer_1']
        layer_2 = latent['layer_2']
        
        layer_1_1 = tf.tile(layer_1, [tf.shape(x)[0],1,1,1])
        layer_2_1 = tf.tile(layer_2, [tf.shape(x)[0],1,1,1])

        x_1 = tf.tile(tf.expand_dims(tf.expand_dims(x,2), 1), [1, tf.shape(layer_1)[1], 1, 1])
        x_2 = tf.concat(2, [x_1, tf.ones((tf.shape(x_1)[0], tf.shape(x_1)[1], 1, 1))])

        l_1 = tf.batch_matmul(layer_1_1, x_2) / tf.sqrt(tf.cast(tf.shape(x_2)[2], tf.float32))
        l_2 = tf.concat(2, [l_1, tf.ones((tf.shape(l_1)[0], tf.shape(l_1)[1], 1, 1))])
        l_3 = tf.nn.relu(l_2)

        y = tf.batch_matmul(layer_2_1, l_3) / tf.sqrt(tf.cast(tf.shape(l_3)[2], tf.float32))
        return y

    def rmse(self, latent, observed, std_y_train = 1):
        """
        calculate the rmse.
        """
        network_output = self.get_output_for(latent, observed['x'])
        network_output_1 = tf.squeeze(network_output, [3])
        network_output_mean = tf.reduce_mean(network_output_1, 1)

        rmse = tf.sqrt(tf.reduce_mean((network_output_mean - observed['y'])**2)) * std_y_train
        return rmse

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

if __name__ == '__main__':
    tf.set_random_seed(1237)
    np.random.seed(1234)
    
    x_train, y_train, x_valid, y_valid, x_test, y_test = get_data('data/2concrete.txt')
    x_train = np.vstack([x_train, x_valid]).astype('float32')
    y_train = np.vstack([y_train.reshape((len(y_train),1)), y_valid.reshape((len(y_valid),1))]).astype('float32')

    x_test = x_test.astype('float32')
    n_x = x_train.shape[1]
    print(x_train.shape[0])

    std_x_train = np.std(x_train, 0)
    std_x_train[ std_x_train == 0 ] = 1
    mean_x_train = np.mean(x_train, 0)
    x_train = (x_train - np.full(x_train.shape, mean_x_train,dtype='float32')) / \
			np.full(x_train.shape, std_x_train,dtype='float32')
    x_test = (x_test - np.full(x_test.shape, mean_x_train,dtype='float32')) / \
			np.full(x_test.shape, std_x_train,dtype='float32')
    mean_y_train = np.mean(y_train)
    std_y_train = np.std(y_train)
    y_train = (y_train - mean_y_train) / std_y_train
    y_test = (y_test - mean_y_train) / std_y_train
    #mean_y_train = 0
    #std_y_train = 1

    # Define training/evaluation parameters
    lb_samples = 10
    ll_samples = 500
    epoches = 500
    batch_size = 10
    test_batch_size = 10
    iters = int(np.floor(x_train.shape[0] / float(batch_size)))
    test_iters = int(np.floor(x_test.shape[0] / float(test_batch_size)))
    test_freq = 10
    learning_rate = 0.01
    anneal_lr_freq = 100
    anneal_lr_rate = 0.75

    #build training model
    model = Model()
    learning_rate_ph = tf.placeholder(tf.float32, shape=())
    observed_x = tf.placeholder(tf.float32, shape=(None, x_train.shape[1]))
    observed_y = tf.placeholder(tf.float32, shape=(None, 1))
    observed = {'x':observed_x, 'y':observed_y}
    n_samples = tf.placeholder(tf.int32, shape=())
    optimizer = tf.train.AdamOptimizer(learning_rate_ph, epsilon=1e-4)

    #shape is (batch_size, sample_num, shape_parameter)
    mu_1 = tf.Variable(tf.zeros([1, 1, 50, x_train.shape[1] + 1]))
    logvar_1 = tf.Variable(tf.ones([1, 1, 50, x_train.shape[1] + 1]))
    mu_2 = tf.Variable(tf.zeros([1, 1, 1, 50 + 1]))
    logvar_2 = tf.Variable(tf.ones([1, 1, 1, 50 + 1]))

    #build q
    layer_mu_1 = InputLayer((1, 1, 50, x_train.shape[1] + 1), input=mu_1)
    layer_logvar_1 = InputLayer((1, 1, 50, x_train.shape[1] + 1), input=logvar_1)
    layer_mu_2 = InputLayer((1, 1, 1, 50 + 1), input=mu_2)
    layer_logvar_2 = InputLayer((1, 1, 1, 50 + 1), input=logvar_2)
    layer_1 = ReparameterizedNormal([layer_mu_1, layer_logvar_1], n_samples)
    layer_2 = ReparameterizedNormal([layer_mu_2, layer_logvar_2], n_samples)
    latent = {'layer_1':get_output(layer_1), 'layer_2':get_output(layer_2)}

    lower_bound = tf.reduce_mean(advi(model, observed, latent, reduction_indices=1))
    grads = optimizer.compute_gradients(-lower_bound)
    infer = optimizer.apply_gradients(grads)
    #infer = optimizer.minimize(-lower_bound)

    latent_outputs = {'layer_1':latent['layer_1'][0], 'layer_2':latent['layer_2'][0]}
    rmse = model.rmse(latent_outputs, observed, std_y_train)
    output = tf.reduce_mean(model.get_output_for(latent_outputs, observed_x), 1)

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
            lbs = []
            for t in range(iters):
                x_batch = x_train[t * batch_size:(t + 1) * batch_size]
                y_batch = y_train[t * batch_size:(t + 1) * batch_size].reshape((batch_size, 1))

                _, grad1, lb= sess.run(
                    [infer, grads, lower_bound],
                        feed_dict={n_samples: lb_samples, learning_rate_ph:learning_rate,
                                   observed_x:x_batch, observed_y:y_batch})

                lbs.append(lb)
            time_epoch += time.time()
            print('Epoch {} ({:.1f}s): Lower bound = {}'.format(
                epoch, time_epoch, np.mean(lbs)))
            #pdb.set_trace()
            if epoch % test_freq == 0:
                time_test = -time.time()
                test_lbs = []
                test_rmses = []
                for t in range(test_iters):
                    test_x_batch = x_test[t * test_batch_size: (t + 1) * test_batch_size]
                    test_y_batch = y_test[t * test_batch_size: (t + 1) * test_batch_size].reshape((test_batch_size, 1))
                    test_lb, test_rmse = sess.run(
                        [lower_bound, rmse],
                            feed_dict={n_samples: ll_samples, observed_x:test_x_batch, observed_y:test_y_batch})

                    test_lbs.append(test_lb)
                    test_rmses.append(test_rmse)
                time_test += time.time()
                #pdb.set_trace()
                out1 = sess.run(output, feed_dict={observed_x: x_test[0:5], n_samples: ll_samples})
                print('>>> TEST ({:.1f}s)'.format(time_test))
                print('>> Test lower bound = {}'.format(np.mean(test_lbs)))
                print('>> Test rmse = {}'.format(np.mean(test_rmses)))
                print(out1.reshape((1,5)))
                print(y_test[0:5])
