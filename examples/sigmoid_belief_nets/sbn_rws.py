#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import time

import tensorflow as tf
from six.moves import range, zip
import numpy as np
import zhusuan as zs

from examples import conf
from examples.utils import dataset


@zs.reuse('model')
def sbn(observed, n, n_x, n_h, n_particles):
    with zs.BayesianNet(observed=observed) as model:
        h3_logits = tf.zeros([n, n_h])
        h3 = zs.Bernoulli('h3', h3_logits, n_samples=n_particles,
                          group_ndims=1, dtype=tf.float32)
        h2_logits = tf.layers.dense(h3, n_h)
        h2 = zs.Bernoulli('h2', h2_logits, group_ndims=1, dtype=tf.float32)
        h1_logits = tf.layers.dense(h2, n_h)
        h1 = zs.Bernoulli('h1', h1_logits, group_ndims=1, dtype=tf.float32)
        x_logits = tf.layers.dense(h1, n_x)
        x = zs.Bernoulli('x', x_logits, group_ndims=1)
    return model


@zs.reuse('proposal')
def q_net(x, n_h, n_particles):
    with zs.BayesianNet() as proposal:
        h1_logits = tf.layers.dense(tf.to_float(x), n_h)
        h1 = zs.Bernoulli('h1', h1_logits, n_samples=n_particles,
                          group_ndims=1, dtype=tf.float32)
        h2_logits = tf.layers.dense(h1, n_h)
        h2 = zs.Bernoulli('h2', h2_logits, group_ndims=1, dtype=tf.float32)
        h3_logits = tf.layers.dense(h2, n_h)
        h3 = zs.Bernoulli('h3', h3_logits, group_ndims=1, dtype=tf.float32)
    return proposal


def main():
    tf.set_random_seed(1237)
    np.random.seed(1234)

    # Load MNIST
    data_path = os.path.join(conf.data_dir, 'mnist.pkl.gz')
    x_train, t_train, x_valid, t_valid, x_test, t_test = \
        dataset.load_mnist_realval(data_path)
    x_train = np.vstack([x_train, x_valid]).astype('float32')
    x_test = np.random.binomial(1, x_test, size=x_test.shape).astype('float32')
    n_x = x_train.shape[1]

    # Define model parameters
    n_h = 200

    # Define training/evaluation parameters
    lb_samples = 10
    ll_samples = 1000
    epochs = 3000
    batch_size = 24
    test_batch_size = 100
    iters = x_train.shape[0] // batch_size
    test_iters = x_test.shape[0] // test_batch_size
    test_freq = 10
    learning_rate = 0.001
    anneal_lr_freq = 200
    anneal_lr_rate = 0.75

    # Build the computation graph
    n_particles = tf.placeholder(tf.int32, shape=[], name='n_particles')

    x_orig = tf.placeholder(tf.float32, shape=[None, n_x], name='x')
    x_bin = tf.cast(tf.less(tf.random_uniform(tf.shape(x_orig), 0, 1), x_orig),
                    tf.int32)
    x = tf.placeholder(tf.int32, shape=[None, n_x], name='x')
    x_obs = tf.tile(tf.expand_dims(x, 0), [n_particles, 1, 1])
    n = tf.shape(x)[0]
    h_names = ['h' + str(i + 1) for i in range(3)]

    def log_joint(observed):
        model = sbn(observed, n, n_x, n_h, n_particles)
        log_phs = model.local_log_prob(h_names)
        log_px_h1 = model.local_log_prob('x')
        return tf.add_n(log_phs) + log_px_h1

    proposal = q_net(x, n_h, n_particles)
    qh_outputs = proposal.query(h_names, outputs=True, local_log_prob=True)
    latent = dict(zip(h_names, qh_outputs))

    learning_rate_ph = tf.placeholder(tf.float32, shape=[], name='lr')
    optimizer = tf.train.AdamOptimizer(learning_rate_ph, epsilon=1e-4)

    # learning model parameters
    lower_bound = tf.reduce_mean(
        zs.variational.importance_weighted_objective(
            log_joint, observed={'x': x_obs}, latent=latent, axis=0))
    model_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                     scope='model')
    model_grads = optimizer.compute_gradients(-lower_bound, model_params)

    # adapting the proposal
    klpq_obj = zs.variational.klpq(
        log_joint, observed={'x': x_obs}, latent=latent, axis=0)
    klpq_cost = tf.reduce_mean(klpq_obj.rws())
    proposal_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                        scope='proposal')
    klpq_grads = optimizer.compute_gradients(klpq_cost, proposal_params)

    infer_op = optimizer.apply_gradients(model_grads + klpq_grads)

    # Run the inference
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(1, epochs + 1):
            time_epoch = -time.time()
            if epoch % anneal_lr_freq == 0:
                learning_rate *= anneal_lr_rate
            np.random.shuffle(x_train)
            lbs = []
            for t in range(iters):
                x_batch = x_train[t * batch_size:(t + 1) * batch_size]
                x_batch_bin = sess.run(x_bin, feed_dict={x_orig: x_batch})
                _, lb = sess.run([infer_op, lower_bound],
                                 feed_dict={x: x_batch_bin,
                                            learning_rate_ph: learning_rate,
                                            n_particles: lb_samples})
                lbs.append(lb)
            time_epoch += time.time()
            print('Epoch {} ({:.1f}s): Lower bound = {}'.format(
                epoch, time_epoch, np.mean(lbs)))

            if epoch % test_freq == 0:
                time_test = -time.time()
                test_lbs = []
                test_lls = []
                for t in range(test_iters):
                    test_x_batch = x_test[
                        t * test_batch_size: (t + 1) * test_batch_size]
                    test_lb = sess.run(lower_bound,
                                       feed_dict={x: test_x_batch,
                                                  n_particles: lb_samples})
                    test_ll = sess.run(lower_bound,
                                       feed_dict={x: test_x_batch,
                                                  n_particles: ll_samples})
                    test_lbs.append(test_lb)
                    test_lls.append(test_ll)
                time_test += time.time()
                print('>>> TEST ({:.1f}s)'.format(time_test))
                print('>> Test lower bound = {}'.format(np.mean(test_lbs)))
                print('>> Test log likelihood = {}'.format(np.mean(test_lls)))


if __name__ == "__main__":
    main()
