# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import sys
import os
import time

import tensorflow as tf
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#from matplotlib import pyplot as plt
import scipy
import scipy.stats
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from zhusuan.model import *
except:
    raise ImportError()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import dataset
except:
    raise ImportError()


tf.set_random_seed(1)


# Test gaussian
num_samples = 10
#data1 = np.random.randn((num_samples))
data1 = np.zeros((num_samples))
tdata = tf.constant(data1, dtype=tf.float32)
with StochasticGraph() as model:
    lx = Normal(tf.zeros((num_samples)), tf.zeros((num_samples)))

random_x, p_random_x = model.get_output(lx)
_, p_x = model.get_output(lx, inputs={lx: tdata})

sess = tf.Session()
params = tf.trainable_variables()
for i in params:
    print(i.name, i.get_shape())
sess.run(tf.global_variables_initializer())

# Check pdf
rx, px = sess.run([random_x, p_random_x])
print('logpdf difference = {}'.format(px - np.log(scipy.stats.norm.pdf(rx))))
print('logpdf difference = {}'.format(sess.run(p_x) - np.log(scipy.stats.norm.pdf(data1))))


# Test BLR
# Load MNIST dataset
n = 600
n_dims = 784
mu = 0
sigma = 1. / math.sqrt(n)

data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'data', 'mnist.pkl.gz')
X_train, y_train, _, _, X_test, y_test = \
    dataset.load_binary_mnist_realval(data_path)
X_train = X_train[:n] * 256
y_train = y_train[:n]
X_test = X_test * 256

def blrf(X_train, y_train, beta, sigma):
    D = len(beta)
    l = D * (-0.5 * np.log(2*np.pi) - np.log(sigma))
    l -= 0.5/sigma**2 * np.sum(np.square(beta))
    a = np.sum(X_train * beta, 1)
    a[a < -30] = -30
    a[a > 30] = 30
    h = 1 / (1 + np.exp(-a))
    # print('Prior = {}'.format(l))
    # print('Likelihood = {}'.format(np.sum(y_train * np.log(h)) + np.sum((1-y_train) * np.log(1-h))))
    l += np.sum(y_train * np.log(h)) + np.sum((1-y_train) * np.log(1-h))
    grad = -beta/sigma**2 + np.sum(X_train.transpose()*(y_train*(1-h)), 1) \
           - np.sum(X_train.transpose()*((1-y_train)*h), 1)

    accuracy = np.mean((a>0.5)==y_train)
    return l, accuracy, grad

def numerical_gradient(func, x):
    grad = np.zeros(x.shape)
    epsilon = 1e-5
    for i in range(len(x)):
        x[i] += epsilon
        p = func(x)
        x[i] -= 2 * epsilon
        n = func(x)
        x[i] += epsilon
        grad[i] = (p - n) / (2 * epsilon)
    return grad

def get_grad(x):
    l, _, _ = blrf(X_train, y_train, x, sigma)
    return l

# Data
x_input = tf.placeholder(tf.float32, [None, n_dims], name='x_input')
x = tf.Variable(tf.zeros((n, n_dims)), trainable=False, name='x')
y = tf.placeholder(tf.float32, [None], name='y')
update_data = tf.assign(x, x_input, validate_shape=False, name='update_data')

class BLR:
    def __init__(self, x):
        with StochasticGraph() as model:
            beta = Normal(tf.zeros((n_dims)), tf.ones((n_dims)) * tf.log(sigma))
            h = tf.reduce_sum(x * beta.value, 1)
            y_mean = tf.sigmoid(h)
            y = Bernoulli(h)

        self.model = model
        self.y = y
        self.beta = beta
        self.y_mean = y_mean
        self.h = h

    def log_prob(self, latent, observed, given):
        # p(y, beta | X)
        y = observed['y']
        beta = latent['beta']
        y_out, beta_out = self.model.get_output([self.y, self.beta],
                                                inputs={self.y: y, self.beta: tf.identity(beta)})

        return tf.reduce_sum(y_out[1]) + tf.reduce_sum(beta_out[1])

beta_0 = np.random.randn(n_dims) * 1e-3
beta = tf.constant(beta_0, dtype=tf.float32)
blr = BLR(x)
l_star, _, grad_star = blrf(X_train, y_train, beta_0, sigma)
l = blr.log_prob({'beta': beta}, {'y': y}, None)
grad = tf.gradients(l, beta)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(update_data, feed_dict={x_input: X_train})
l_est, grad_est = sess.run([l, grad], feed_dict={y: y_train})

num_grad = numerical_gradient(get_grad, beta_0)

print('Likelihood difference = {}'.format((l_star - l_est)/l_star))
print('Gradient difference = {}'.format(np.sqrt(np.sum(np.square(grad_star - grad_est)))/np.square(np.sum(np.square(grad_star)))))
print('Gradient difference = {}'.format(np.sqrt(np.sum(np.square(grad_star - num_grad)))/np.square(np.sum(np.square(grad_star)))))
#print(grad_est)
#print(grad_star*2 - grad_est)

        # def log_posterior(latent, observed, given):
#     _, log_p = model.get_output(lx, inputs={lx: tf.identity(latent['x'])})
#     return log_p
#
# hmc = HMC(step_size=0.3, n_leapfrogs=5)
#
# x = tf.Variable(tf.zeros((num_chains)), name='x')
# sampler = hmc.sample(log_posterior, {}, {'x': x}, chain_axis=0)
#
# sess = tf.Session()
# params = tf.trainable_variables()
# for i in params:
#     print(i.name, i.get_shape())
# sess.run(tf.global_variables_initializer())
#
# train_writer = tf.summary.FileWriter('train', tf.get_default_graph())
# train_writer.close()
#
# samples = []
# print('Sampling...')
# for i in range(num_samples):
#     q, p, oh, nh, ll, ar = sess.run(sampler)
#     #print(q, p, oh, nh, ar)
#     if isinstance(q[0], np.ndarray):
#         samples.extend(list(q[0]))
#     else:
#         samples.append(q[0])
# print('Finished.')
#
#
# def kde(xs, mu, batch_size):
#     mu_n = len(mu)
#     assert(mu_n % batch_size == 0)
#     xs_row = np.expand_dims(xs, 1)
#     ys = np.zeros(xs.shape)
#
#     for b in range(mu_n // batch_size):
#         mu_col = np.expand_dims(mu[b*batch_size:(b+1)*batch_size], 0)
#         ys += (1 / np.sqrt(2 * np.pi) / kernel_width) * \
#              np.mean(np.exp((-0.5 / kernel_width ** 2) * np.square(xs_row - mu_col)), 1)
#
#     ys /= (mu_n / batch_size)
#     return ys
#
# xs = np.linspace(-5, 5, 1000)
# ys = kde(xs, np.array(samples), num_chains)
#
# f, ax = plt.subplots()
# ax.plot(xs, ys)
# ax.plot(xs, scipy.stats.norm.pdf(xs))
#
# #print(samples)
# print(scipy.stats.normaltest(samples))
#
# plt.show()