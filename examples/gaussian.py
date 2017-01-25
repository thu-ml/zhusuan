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
from zhusuan.mcmc.hmc import HMC
from matplotlib import pyplot as plt
import scipy
import scipy.stats
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import zhusuan as zs
except:
    raise ImportError()

tf.set_random_seed(1)

kernel_width = 0.1
num_samples = 100
num_chains = 1000
burnin = num_samples // 2
n_dims = 10
stdev = np.abs(np.random.rand(n_dims) + 1e-3)
print(stdev)
log_stdev = np.log(stdev)

def gaussian(observed):
    with zs.StochasticGraph(observed=observed) as model:
        lx = zs.Normal('x', tf.zeros((num_chains, n_dims)),
                       np.tile(log_stdev, (num_chains, 1)))
    return model


def log_joint(latent, observed, given):
    model = gaussian(latent)
    log_p = model.local_log_prob(['x'])
    return tf.reduce_sum(log_p[0], -1)

adapt_step_size = tf.placeholder(dtype=tf.bool, shape=[], name="adapt_step_size")
hmc = HMC(step_size=0.3, n_leapfrogs=1, adapt_step_size=adapt_step_size)
#hmc = HMC(step_size=0.3, n_leapfrogs=1)

x = tf.Variable(tf.zeros((num_chains, n_dims)), name='x')
sampler = hmc.sample(log_joint, {}, {'x': x}, chain_axis=0)

sess = tf.Session()
params = tf.trainable_variables()
for i in params:
    print(i.name, i.get_shape())
sess.run(tf.global_variables_initializer())

train_writer = tf.summary.FileWriter('train', tf.get_default_graph())
train_writer.close()

samples = []
print('Sampling...')
for i in range(num_samples):
    q, p, oh, nh, ol, nl, ar, ss = sess.run(sampler,
                                            feed_dict={adapt_step_size: i < burnin})
    print(np.mean(ar), ss)
    #print(q, p, oh, nh, ar)

    if i >= burnin:
        samples.append(q[0])
print('Finished.')
samples = np.vstack(samples)


def kde(xs, mu, batch_size):
    mu_n = len(mu)
    assert(mu_n % batch_size == 0)
    xs_row = np.expand_dims(xs, 1)
    ys = np.zeros(xs.shape)

    for b in range(mu_n // batch_size):
        mu_col = np.expand_dims(mu[b*batch_size:(b+1)*batch_size], 0)
        ys += (1 / np.sqrt(2 * np.pi) / kernel_width) * \
             np.mean(np.exp((-0.5 / kernel_width ** 2) * np.square(xs_row - mu_col)), 1)

    ys /= (mu_n / batch_size)
    return ys

if n_dims == 1:
    xs = np.linspace(-5, 5, 1000)
    ys = kde(xs, np.squeeze(samples), num_chains)

    f, ax = plt.subplots()
    ax.plot(xs, ys)
    ax.plot(xs, scipy.stats.norm.pdf(xs, scale=stdev[0]))

for i in range(n_dims):
    print(scipy.stats.normaltest(samples[:,i]))

print('Sample mean = {}'.format(np.mean(samples, 0)))
print('Expected stdev = {}'.format(np.std(samples, 0)))
print('Relative error of stdev = {}'.format((np.std(samples, 0)-stdev)/stdev))

plt.show()
