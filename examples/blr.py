#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import math
from dataset import load_german_credits
from zhusuan.optimization.gradient_descent_optimizer import \
    GradientDescentOptimizer
from zhusuan.mcmc.nuts import NUTS

float_eps = 1e-30

# Load MNIST dataset
# n = 600
# n_dims = 784
# mu = 0
# sigma = 1./math.sqrt(n)
#
# X_train, y_train, X_test, y_test, beta_map, _, beta_n,
#  _ = lr_datasets.binary_mnist_600(n, sigma, intercept=False)

# Load German credits dataset
n = 900
n_dims = 24
mu = 0
sigma = 1./math.sqrt(n)

X_train, y_train, X_test, y_test = load_german_credits(n)

# Define graph
# Data
x_input = tf.placeholder(tf.float32, [None, n_dims], name='x_input')
x = tf.Variable(tf.zeros((n, n_dims)), trainable=False, name='x')
y = tf.placeholder(tf.float32, [None], name='y')
update_data = tf.assign(x, x_input, validate_shape=False, name='update_data')

# Model
beta = tf.Variable(np.zeros((n_dims)), dtype=tf.float32, name='beta')
scores = tf.reduce_sum(x*beta, reduction_indices=(1))
logits = tf.nn.sigmoid(scores, name='logits')
predictions = tf.cast(logits > 0.5, tf.float32)
n_correct = tf.reduce_sum(predictions * y + (1-predictions) * (1-y))
loss = -tf.reduce_sum(y * tf.log(logits + float_eps) +
                      (1-y) * tf.log(1-logits + float_eps), name='loss')
log_prior = -tf.reduce_sum(tf.square(beta-mu) / (2*sigma**2))
log_likelihood = tf.sub(log_prior, loss, name='log_likelihood')
vars = [beta]

sess = tf.Session()
train_writer = tf.train.SummaryWriter('train', sess.graph)

# Find a MAP solution
sess.run(tf.initialize_all_variables())
sess.run(update_data, feed_dict={x_input: X_train})

optimizer = GradientDescentOptimizer(sess, {y: y_train}, -log_likelihood,
                                     vars, stepsize_tol=1e-9, tol=1e-5)
optimizer.optimize()

chain_length = 100
burnin = 50
# sampler = NUTS(sess, {y: y_train}, log_likelihood, 1e-2, mass=mass,
# m_adapt=burnin, mass_adaptation=True, burnin=burnin)
# sampler = NUTS(sess, {y: y_train}, log_likelihood, 1e-2,
# mass_adaptation=True, m_adapt=burnin, burnin=burnin)
sampler = NUTS(sess, {y: y_train}, [beta], log_likelihood, 1e-2,
               m_adapt=burnin)
sample_sum = []
num_samples = chain_length - burnin
train_scores = np.zeros((X_train.shape[0]))
test_scores = np.zeros((X_test.shape[0]))
for i in range(chain_length):
    # Feed data in
    sess.run(update_data, feed_dict={x_input: X_train})
    model = sampler.sample()
    if i == burnin:
        sample_sum = model
    elif i > burnin:
        for j in range(len(model)):
            sample_sum[j] += model[j]

    # print(lr(X_train, y_train, model[0], model[1], sigma**2)[:2])

    # evaluate
    n_train_c, train_pred_c, ll = sess.run(
        (n_correct, logits, log_likelihood), feed_dict={y: y_train})
    sess.run(update_data, feed_dict={x_input: X_test})
    n_test_c, test_pred_c = sess.run((n_correct, logits),
                                     feed_dict={y: y_test})
    print('Log likelihood = %f, Train set accuracy = %f, '
          'test set accuracy = %f' %
          (ll, (float(n_train_c) / X_train.shape[0]),
           (float(n_test_c) / X_test.shape[0])))
    # print(sess.run((beta, bias)))

    if i >= burnin:
        train_scores += train_pred_c
        test_scores += test_pred_c

train_scores /= num_samples
test_scores /= num_samples

set_mean = []
for j in range(len(vars)):
    set_mean.append(vars[j].assign(sample_sum[j] / num_samples))
sess.run(set_mean)

sess.run(update_data, feed_dict={x_input: X_train})
r_loss = sess.run(loss, feed_dict={y: y_train})
n_train_c = sess.run(n_correct, feed_dict={y: y_train})
sess.run(update_data, feed_dict={x_input: X_test})
n_test_c = sess.run(n_correct, feed_dict={y: y_test})

train_pred = (train_scores > 0.5).astype(np.float32)
test_pred = (test_scores > 0.5).astype(np.float32)

train_accuracy = float(np.sum(train_pred == y_train)) / X_train.shape[0]
test_accuracy = float(np.sum(test_pred == y_test)) / X_test.shape[0]

print('Log likelihood of expected parameters: %f, train set accuracy = %f, '
      'test set accuracy = %f' %
      (r_loss, (float(n_train_c) / X_train.shape[0]),
       (float(n_test_c) / X_test.shape[0])))
print('Gibbs classifier: train set accuracy = %f, test set accuracy = %f'
      % (train_accuracy, test_accuracy))

sampler.stat(burnin)
