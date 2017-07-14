#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import time

import tensorflow as tf
from tensorflow.contrib import layers
from six.moves import range
import numpy as np
import zhusuan as zs

from examples import conf
from examples.utils import dataset, save_image_collections


temperature_prior = 0.5
temperature_posterior = 0.666


@zs.reuse('model')
def vae(observed, n, n_x, n_z, n_k, n_particles, is_training, relaxed):
    with zs.BayesianNet(observed=observed) as model:
        normalizer_params = {'is_training': is_training,
                             'updates_collections': None}
        z_logits = tf.get_variable('z_logits', [n_z * n_k])
        z_logits = tf.tile(tf.expand_dims(z_logits, 0), [n, 1])
        z_stacked_logits = tf.reshape(z_logits, [n, n_z, n_k])
        if relaxed:
            z = zs.ExpConcrete('z', temperature_prior, z_stacked_logits,
                               n_samples=n_particles, group_event_ndims=1)
            z = tf.exp(tf.reshape(z, [n_particles, n, n_z * n_k]))
        else:
            z = zs.OnehotCategorical('z', z_stacked_logits,
                                     n_samples=n_particles,
                                     group_event_ndims=1)
            z = tf.to_float(tf.reshape(z, [n_particles, n, n_z * n_k]))
        lx_z = layers.fully_connected(
            z, 500, activation_fn=tf.tanh,
            normalizer_fn=layers.batch_norm,
            normalizer_params=normalizer_params)
        lx_z = layers.fully_connected(
            lx_z, 500, activation_fn=tf.tanh,
            normalizer_fn=layers.batch_norm,
            normalizer_params=normalizer_params)
        x_logits = layers.fully_connected(lx_z, n_x, activation_fn=None)
        x = zs.Bernoulli('x', x_logits, group_event_ndims=1)
    return model


@zs.reuse('variational')
def q_net(observed, x, n_z, n_k, n_particles, is_training, relaxed):
    with zs.BayesianNet(observed=observed) as variational:
        normalizer_params = {'is_training': is_training,
                             'updates_collections': None}
        lz_x = layers.fully_connected(
            tf.to_float(x), 500, activation_fn=tf.tanh,
            normalizer_fn=layers.batch_norm,
            normalizer_params=normalizer_params)
        lz_x = layers.fully_connected(
            lz_x, 500, activation_fn=tf.tanh,
            normalizer_fn=layers.batch_norm,
            normalizer_params=normalizer_params)
        z_logits = layers.fully_connected(lz_x, n_z * n_k, activation_fn=None)
        z_stacked_logits = tf.reshape(z_logits, [n, n_z, n_k])
        if relaxed:
            z = zs.ExpConcrete('z', temperature_posterior, z_stacked_logits,
                               n_samples=n_particles, group_event_ndims=1)
        else:
            z = zs.OnehotCategorical('z', z_stacked_logits,
                                     n_samples=n_particles,
                                     group_event_ndims=1)
    return variational


if __name__ == '__main__':
    tf.set_random_seed(1237)
    np.random.seed(1237)

    # Load MNIST
    data_path = os.path.join(conf.data_dir, 'mnist.pkl.gz')
    x_train, t_train, x_valid, t_valid, x_test, t_test = \
        dataset.load_mnist_realval(data_path)
    x_train = np.vstack([x_train, x_valid]).astype('float32')
    x_test = np.random.binomial(1, x_test, size=x_test.shape).astype('float32')

    # Define parameters
    n_z, n_k = 100, 2   # number of latent variables, categories
    n_x = x_train.shape[1]
    lb_samples = 1
    ll_samples = 500
    epochs = 10000
    batch_size = 100
    iters = x_train.shape[0] // batch_size
    learning_rate = 0.001
    anneal_lr_freq = 200
    anneal_lr_rate = 1.0
    test_freq = 50
    test_batch_size = 400
    test_iters = x_test.shape[0] // test_batch_size
    save_freq = 20
    result_path = "results/vae"

    # Build the computation graph
    is_training = tf.placeholder(tf.bool, shape=[], name='is_training')
    n_particles = tf.placeholder(tf.int32, shape=[], name='n_particles')
    x_orig = tf.placeholder(tf.float32, shape=[None, n_x], name='x')
    x_bin = tf.cast(tf.less(tf.random_uniform(tf.shape(x_orig), 0, 1), x_orig),
                    tf.int32)
    x = tf.placeholder(tf.int32, shape=[None, n_x], name='x')
    x_obs = tf.tile(tf.expand_dims(x, 0), [n_particles, 1, 1])
    n = tf.shape(x)[0]

    def lower_bound_and_log_likelihood(relaxed):
        def log_joint(observed):
            model = vae(observed, n, n_x, n_z, n_k,
                        n_particles, is_training, relaxed)
            log_pz, log_px_z = model.local_log_prob(['z', 'x'])
            return log_pz + log_px_z

        variational = q_net({}, x, n_z, n_k, n_particles, is_training, relaxed)
        qz_samples, log_qz = variational.query('z', outputs=True,
                                               local_log_prob=True)
        lower_bound = tf.reduce_mean(zs.sgvb(
            log_joint, {'x': x_obs}, {'z': [qz_samples, log_qz]}, axis=0))

        # Importance sampling estimates of marginal log likelihood
        is_log_likelihood = tf.reduce_mean(
            zs.is_loglikelihood(log_joint, {'x': x_obs},
                                {'z': [qz_samples, log_qz]}, axis=0))

        return lower_bound, is_log_likelihood

    # For training
    relaxed_lower_bound, _ = lower_bound_and_log_likelihood(True)
    # For testing and generating
    lower_bound, is_log_likelihood = lower_bound_and_log_likelihood(False)

    learning_rate_ph = tf.placeholder(tf.float32, shape=[], name='lr')
    optimizer = tf.train.AdamOptimizer(learning_rate_ph, epsilon=1e-4)
    grads = optimizer.compute_gradients(-relaxed_lower_bound)
    infer = optimizer.apply_gradients(grads)

    # Generate images
    n_gen = 100
    model = vae({}, n_gen, n_x, n_z, n_k, 1, is_training, False)
    x_logits = model._stochastic_tensors['x'].distribution.logits
    x_gen = tf.reshape(tf.sigmoid(x_logits), [-1, 28, 28, 1])

    params = tf.trainable_variables()
    for i in params:
        print(i.name, i.get_shape())

    saver = tf.train.Saver(max_to_keep=10)

    # Run the inference
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Restore from the latest checkpoint
        ckpt_file = tf.train.latest_checkpoint(result_path)
        begin_epoch = 1
        if ckpt_file is not None:
            print('Restoring model from {}...'.format(ckpt_file))
            begin_epoch = int(ckpt_file.split('.')[-2]) + 1
            saver.restore(sess, ckpt_file)

        for epoch in range(begin_epoch, epochs + 1):
            time_epoch = -time.time()
            if epoch % anneal_lr_freq == 0:
                learning_rate *= anneal_lr_rate
            np.random.shuffle(x_train)
            lbs = []
            for t in range(iters):
                x_batch = x_train[t * batch_size:(t + 1) * batch_size]
                x_batch_bin = sess.run(x_bin, feed_dict={x_orig: x_batch})
                _, lb = sess.run([infer, relaxed_lower_bound],
                                 feed_dict={x: x_batch_bin,
                                            learning_rate_ph: learning_rate,
                                            n_particles: lb_samples,
                                            is_training: True})
                lbs.append(lb)
            time_epoch += time.time()
            print('Epoch {} ({:.1f}s): Lower bound = {}'.format(
                epoch, time_epoch, np.mean(lbs)))

            if epoch % test_freq == 0:
                time_test = -time.time()
                test_lbs = []
                test_lls = []
                for t in range(test_iters):
                    test_x_batch = x_test[t * test_batch_size:
                                          (t + 1) * test_batch_size]
                    test_lb = sess.run(lower_bound,
                                       feed_dict={x: test_x_batch,
                                                  n_particles: ll_samples,
                                                  is_training: False})
                    test_ll = sess.run(is_log_likelihood,
                                       feed_dict={x: test_x_batch,
                                                  n_particles: ll_samples,
                                                  is_training: False})
                    test_lbs.append(test_lb)
                    test_lls.append(test_ll)
                time_test += time.time()
                print('>>> TEST ({:.1f}s)'.format(time_test))
                print('>> Test lower bound = {}'.format(np.mean(test_lbs)))
                print('>> Test log likelihood (IS) = {}'.format(
                    np.mean(test_lls)))

            if epoch % save_freq == 0:
                print('Saving model...')
                save_path = os.path.join(result_path,
                                         "vae.epoch.{}.ckpt".format(epoch))
                if not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path))
                saver.save(sess, save_path)

                images = sess.run(x_gen, feed_dict={is_training: False})
                name = "results/vae/vae.epoch.{}.png".format(epoch)
                save_image_collections(images, name)
                print('Done')
