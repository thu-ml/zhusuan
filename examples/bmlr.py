
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import math
import os
from dataset import load_uci_german_credits, load_mnist_realval, one_hot
from zhusuan.optimization.gradient_descent_optimizer import \
    GradientDescentOptimizer
from zhusuan.distributions import norm, discrete
from zhusuan.mcmc.nuts import NUTS

float_eps = 1e-30

# Load MNIST dataset
# n = 600
# n_dims = 784
# n_classes = 10
# mu = 0
# sigma = 1./math.sqrt(n)
#
# data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
#                              'data', 'mnist.pkl.gz')
# X_train, y_train, _, _, X_test, y_test = load_mnist_realval(data_path)
# X_train = X_train[:n] * 256
# y_train = y_train[:n]
# X_test = X_test * 256

# Load German credits dataset
n = 900
n_dims = 24
n_classes = 2
mu = 0
sigma = 1./math.sqrt(n)

data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'data', 'german.data-numeric')
X_train, y_train, X_test, y_test = load_uci_german_credits(data_path, n)
y_train = one_hot(y_train.astype(np.int32), 2)
y_test = one_hot(y_test.astype(np.int32), 2)

# Define graph
# Data
x_input = tf.placeholder(tf.float32, [None, n_dims], name='x_input')
x = tf.Variable(tf.zeros((n, n_dims)), trainable=False, name='x')
y = tf.placeholder(tf.float32, [None, n_classes], name='y')
update_data = tf.assign(x, x_input, validate_shape=False, name='update_data')

# Model
beta = tf.Variable(np.zeros((n_dims, n_classes)),
                   dtype=tf.float32, name='beta')
scores = tf.matmul(x, beta)
logits = tf.nn.softmax(scores)
correct = tf.nn.in_top_k(scores, tf.argmax(y, dimension=1), 1)
n_correct = tf.reduce_sum(tf.cast(correct, tf.int32))

log_likelihood = tf.reduce_sum(norm.logpdf(beta, 0, sigma)) + \
                 tf.reduce_sum(discrete.logpdf(y, logits))

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
sampler = NUTS(sess, {y: y_train}, [beta], log_likelihood, 1e-2,
               m_adapt=burnin)
# sampler = NUTS(sess, {y: y_train}, [beta], log_likelihood, 1e-2,
#                mass_adaptation=True, m_adapt=burnin)

sample_sum = []
num_samples = chain_length - burnin
train_scores = np.zeros((X_train.shape[0], n_classes))
test_scores = np.zeros((X_test.shape[0], n_classes))
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
r_log_likelihood = sess.run(log_likelihood, feed_dict={y: y_train})
n_train_c = sess.run(n_correct, feed_dict={y: y_train})
sess.run(update_data, feed_dict={x_input: X_test})
n_test_c = sess.run(n_correct, feed_dict={y: y_test})

train_pred = np.argmax(train_scores, axis=1)
test_pred = np.argmax(test_scores, axis=1)
train_label = np.argmax(y_train, axis=1)
test_label = np.argmax(y_test, axis=1)

train_accuracy = float(np.sum(train_pred == train_label)) / X_train.shape[0]
test_accuracy = float(np.sum(test_pred == test_label)) / X_test.shape[0]

print('Log likelihood of expected parameters: %f, train set accuracy = %f, '
      'test set accuracy = %f' %
      (r_log_likelihood, (float(n_train_c) / X_train.shape[0]),
       (float(n_test_c) / X_test.shape[0])))
print('Gibbs classifier: train set accuracy = %f, test set accuracy = %f'
      % (train_accuracy, test_accuracy))

sampler.stat(burnin)
