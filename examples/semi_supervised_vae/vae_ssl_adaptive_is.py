#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import time

import tensorflow as tf
from six.moves import range
import numpy as np
import zhusuan as zs

from examples import conf
from examples.utils import dataset


@zs.meta_bayesian_net(scope="gen", reuse_variables=True)
def build_gen(n, x_dim, n_class, z_dim, n_particles):
    bn = zs.BayesianNet()
    z_mean = tf.zeros([n, z_dim])
    z = bn.normal("z", z_mean, std=1., group_ndims=1, n_samples=n_particles)
    h_from_z = tf.layers.dense(z, 500)
    y_logits = tf.zeros([n, n_class])
    y = bn.onehot_categorical("y", y_logits)
    h_from_y = tf.layers.dense(tf.cast(y, tf.float32), 500)
    h = tf.nn.relu(h_from_z + h_from_y)
    h = tf.layers.dense(h, 500, activation=tf.nn.relu)
    x_logits = tf.layers.dense(h, x_dim)
    bn.bernoulli("x", x_logits, group_ndims=1)
    return bn


@zs.reuse_variables(scope="qz_xy")
def qz_xy(x, y, z_dim):
    h = tf.layers.dense(tf.cast(tf.concat([x, y], -1), tf.float32), 500,
                        activation=tf.nn.relu)
    h = tf.layers.dense(h, 500, activation=tf.nn.relu)
    z_mean = tf.layers.dense(h, z_dim)
    z_logstd = tf.layers.dense(h, z_dim)
    return z_mean, z_logstd


@zs.reuse_variables(scope="qy_x")
def qy_x(x, n_class):
    h = tf.layers.dense(tf.cast(x, tf.float32), 500, activation=tf.nn.relu)
    h = tf.layers.dense(h, 500, activation=tf.nn.relu)
    y_logits = tf.layers.dense(h, n_class)
    return y_logits


def labeled_proposal(x, y, z_dim, n_particles):
    bn = zs.BayesianNet()
    z_mean, z_logstd = qz_xy(x, y, z_dim)
    bn.normal("z", z_mean, logstd=z_logstd, n_samples=n_particles,
              group_ndims=1, is_reparameterized=False)
    return bn


def unlabeled_proposal(x, n_class, z_dim, n_particles):
    bn = zs.BayesianNet()
    y_logits = qy_x(x, n_class)
    y = bn.onehot_categorical("y", y_logits)
    z_mean, z_logstd = qz_xy(x, y, z_dim)
    bn.normal("z", z_mean, logstd=z_logstd, group_ndims=1,
              is_reparameterized=False, n_samples=n_particles)
    return bn


def main():
    tf.set_random_seed(1234)
    np.random.seed(1234)

    # Load MNIST
    data_path = os.path.join(conf.data_dir, "mnist.pkl.gz")
    x_labeled, t_labeled, x_unlabeled, x_test, t_test = \
        dataset.load_mnist_semi_supervised(data_path, one_hot=True)
    x_test = np.random.binomial(1, x_test, size=x_test.shape)
    n_labeled, x_dim = x_labeled.shape
    n_class = 10

    # Define model parameters
    z_dim = 100
    beta = 1200.

    # Build the computation graph
    n = tf.placeholder(tf.int32, shape=[], name="n")
    n_particles = tf.placeholder(tf.int32, shape=[], name="n_particles")
    model = build_gen(n, x_dim, n_class, z_dim, n_particles)

    # Labeled
    x_labeled_ph = tf.placeholder(tf.float32, shape=[None, x_dim], name="x_l")
    x_labeled = tf.cast(
        tf.less(tf.random_uniform(tf.shape(x_labeled_ph)), x_labeled_ph),
        tf.int32)
    y_labeled_ph = tf.placeholder(tf.int32, shape=[None, n_class], name="y_l")
    proposal = labeled_proposal(x_labeled, y_labeled_ph, z_dim, n_particles)

    # adapting the proposal
    labeled_klpq_obj = zs.variational.klpq(
        model,
        observed={"x": x_labeled, "y": y_labeled_ph},
        variational=proposal,
        axis=0)
    labeled_q_cost = tf.reduce_mean(labeled_klpq_obj.importance())

    # learning model parameters
    labeled_lower_bound = tf.reduce_mean(
        zs.variational.importance_weighted_objective(
            model, observed={'x': x_labeled, 'y': y_labeled_ph},
            variational=proposal, axis=0))

    # Unlabeled
    x_unlabeled_ph = tf.placeholder(tf.float32, shape=[None, x_dim],
                                    name="x_u")
    x_unlabeled = tf.cast(
        tf.less(tf.random_uniform(tf.shape(x_unlabeled_ph)), x_unlabeled_ph),
        tf.int32)
    proposal = unlabeled_proposal(x_unlabeled, n_class, z_dim, n_particles)

    # adapting the proposal
    unlabeled_klpq_obj = zs.variational.klpq(
        model,
        observed={'x': x_unlabeled},
        variational=proposal,
        axis=0)
    unlabeled_q_cost = tf.reduce_mean(unlabeled_klpq_obj.importance())

    # learning model parameters
    unlabeled_lower_bound = tf.reduce_mean(
        zs.variational.importance_weighted_objective(
            model, observed={'x': x_unlabeled}, variational=proposal,
            axis=0))

    # Build classifier
    qy_logits_l = qy_x(x_labeled, n_class)
    qy_l = tf.nn.softmax(qy_logits_l)
    pred_y = tf.argmax(qy_l, 1)
    acc = tf.reduce_sum(
        tf.cast(tf.equal(pred_y, tf.argmax(y_labeled_ph, 1)), tf.float32) /
        tf.cast(tf.shape(x_labeled)[0], tf.float32))
    onehot_cat = zs.distributions.OnehotCategorical(qy_logits_l)
    log_qy_x = onehot_cat.log_prob(y_labeled_ph)
    classifier_cost = -beta * tf.reduce_mean(log_qy_x)

    # Gather gradients
    proposal_cost = labeled_q_cost + unlabeled_q_cost + classifier_cost
    model_cost = -labeled_lower_bound - unlabeled_lower_bound

    optimizer = tf.train.AdamOptimizer(learning_rate=3e-4)
    model_params = tf.trainable_variables(scope="gen")
    model_grads = optimizer.compute_gradients(model_cost,
                                              var_list=model_params)
    proposal_params = (tf.trainable_variables(scope="qy_x") +
                       tf.trainable_variables(scope="qz_xy"))
    proposal_grads = optimizer.compute_gradients(proposal_cost,
                                                 var_list=proposal_params)
    infer_op = optimizer.apply_gradients(model_grads + proposal_grads)

    # Define training/evaluation parameters
    ll_samples = 10
    epochs = 3000
    batch_size = 100
    iters = x_unlabeled.shape[0] // batch_size
    test_freq = 10
    test_batch_size = 100
    test_iters = x_test.shape[0] // test_batch_size

    # Run the inference
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(1, epochs + 1):
            time_epoch = -time.time()
            np.random.shuffle(x_unlabeled)
            lbs_labeled, lbs_unlabeled, train_accs = [], [], []

            for t in range(iters):
                labeled_indices = np.random.randint(0, n_labeled,
                                                    size=batch_size)
                x_labeled_batch = x_labeled[labeled_indices]
                y_labeled_batch = t_labeled[labeled_indices]
                x_unlabeled_batch = x_unlabeled[t * batch_size:
                                                (t + 1) * batch_size]
                _, lb_labeled, lb_unlabeled, train_acc = sess.run(
                    [infer_op, labeled_lower_bound, unlabeled_lower_bound,
                     acc],
                    feed_dict={x_labeled_ph: x_labeled_batch,
                               y_labeled_ph: y_labeled_batch,
                               x_unlabeled_ph: x_unlabeled_batch,
                               n_particles: ll_samples,
                               n: batch_size})
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
                test_lls_labeled, test_lls_unlabeled, test_accs = [], [], []

                for t in range(test_iters):
                    test_x_batch = x_test[
                        t * test_batch_size: (t + 1) * test_batch_size]
                    test_y_batch = t_test[
                        t * test_batch_size: (t + 1) * test_batch_size]
                    test_ll_labeled, test_ll_unlabeled, test_acc = sess.run(
                        [labeled_lower_bound, unlabeled_lower_bound, acc],
                        feed_dict={x_labeled: test_x_batch,
                                   y_labeled_ph: test_y_batch,
                                   x_unlabeled: test_x_batch,
                                   n_particles: ll_samples,
                                   n: test_batch_size})
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


if __name__ == "__main__":
    main()
