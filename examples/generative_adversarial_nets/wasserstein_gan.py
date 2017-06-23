#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import time

import numpy as np
import tensorflow as tf
from six.moves import range
from tensorflow.contrib import layers
import zhusuan as zs

from examples import conf
from examples.utils import dataset, multi_gpu, save_image_collections
from examples.utils.multi_gpu import FLAGS


@zs.reuse('generator')
def generator(observed, n, n_z, is_training):
    with zs.BayesianNet(observed=observed) as generator:
        normalizer_params = {'is_training': is_training,
                             'updates_collections': None}
        z_min = -tf.ones([n, n_z])
        z_max = tf.ones([n, n_z])
        z = zs.Uniform('z', z_min, z_max)
        lx_z = tf.reshape(z, [-1, 1, 1, n_z])
        ngf = 32
        lx_z = layers.conv2d_transpose(lx_z, ngf*4, 3, padding='VALID',
                                       normalizer_fn=layers.batch_norm,
                                       normalizer_params=normalizer_params)
        lx_z = layers.conv2d_transpose(lx_z, ngf*2, 5, padding='VALID',
                                       normalizer_fn=layers.batch_norm,
                                       normalizer_params=normalizer_params)
        lx_z = layers.conv2d_transpose(lx_z, ngf, 5, stride=2,
                                       normalizer_fn=layers.batch_norm,
                                       normalizer_params=normalizer_params)
        lx_z = layers.conv2d_transpose(lx_z, 1, 5, stride=2,
                                       activation_fn=tf.nn.sigmoid)
    return generator, lx_z


@zs.reuse('discriminator')
def discriminator(x, is_training):
    normalizer_params = {'is_training': is_training,
                         'updates_collections': None}
    ndf = 16
    lc_x = layers.conv2d(x, ndf, 5, stride=2,
                         normalizer_fn=layers.batch_norm,
                         normalizer_params=normalizer_params)
    lc_x = layers.conv2d(lc_x, ndf*2, 5, stride=2,
                         normalizer_fn=layers.batch_norm,
                         normalizer_params=normalizer_params)
    lc_x = layers.conv2d(lc_x, ndf*4, 5, padding='VALID',
                         normalizer_fn=layers.batch_norm,
                         normalizer_params=normalizer_params)
    lc_x = layers.flatten(lc_x)
    critic = layers.fully_connected(lc_x, 1, activation_fn=None)
    return critic


if __name__ == "__main__":
    tf.set_random_seed(1234)

    # Load MINST
    data_path = os.path.join(conf.data_dir, 'mnist.pkl.gz')
    x_train, t_train, x_valid, t_valid, x_test, t_test = \
        dataset.load_mnist_realval(data_path)
    n_xl = 28
    n_channels = 1
    x_train = np.vstack([x_train, x_valid]).astype(np.float32).reshape(
        (-1, n_xl, n_xl, n_channels))
    np.random.seed(1234)
    x_test = x_test.astype(np.float32).reshape((-1, n_xl, n_xl, n_channels))

    # Define model parameters
    n_z = 40

    # Define training/evaluation parameters
    lb_samples = 1
    epochs = 1000
    batch_size = 64 * FLAGS.num_gpus
    gen_size = 100
    iters = x_train.shape[0] // batch_size
    print_freq = 100
    test_freq = iters
    save_freq = iters
    learning_rate = 0.0002
    anneal_lr_freq = 200
    anneal_lr_rate = 0.75

    # Build the computation graph
    is_training = tf.placeholder(tf.bool, shape=[], name='is_training')
    x = tf.placeholder(tf.float32, shape=(None, n_xl, n_xl, n_channels),
                       name='x')
    learning_rate_ph = tf.placeholder(tf.float32, shape=[], name='lr')
    optimizer = tf.train.RMSPropOptimizer(learning_rate_ph, decay=0.5)

    def build_tower_graph(x, id_):
        tower_x = x[id_ * tf.shape(x)[0] // FLAGS.num_gpus:
                    (id_ + 1) * tf.shape(x)[0] // FLAGS.num_gpus]
        n = tf.shape(tower_x)[0]
        gen, x_gen = generator(None, n, n_z, is_training)
        x_critic = discriminator(tower_x, is_training)
        x_gen_critic = discriminator(x_gen, is_training)
        gen_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                         scope='generator')
        disc_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                          scope='discriminator')
        disc_loss = -tf.reduce_mean(x_critic - x_gen_critic)
        gen_loss = -tf.reduce_mean(x_gen_critic)
        disc_grads = optimizer.compute_gradients(
            disc_loss, var_list=disc_var_list)
        gen_grads = optimizer.compute_gradients(
            gen_loss, var_list=gen_var_list)
        grads = disc_grads + gen_grads
        return grads, gen_loss, disc_loss

    tower_losses = []
    tower_grads = []
    for i in range(FLAGS.num_gpus):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('tower_%d' % i):
                grads, gen_loss, disc_loss = build_tower_graph(x, i)
                tower_losses.append([gen_loss, disc_loss])
                tower_grads.append(grads)
    gen_loss, disc_loss = multi_gpu.average_losses(tower_losses)
    grads = multi_gpu.average_gradients(tower_grads)
    infer = optimizer.apply_gradients(grads)
    disc_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                      scope='discriminator')
    with tf.control_dependencies([infer]):
        clip_op = tf.group(
            *[var.assign(tf.clip_by_value(var, -0.01, 0.01)) for var
              in disc_var_list])
    _, eval_x_gen = generator(None, gen_size, n_z, is_training)

    gen_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                     scope='generator')
    saver = tf.train.Saver(max_to_keep=10, var_list=gen_var_list)

    # Run the inference
    with multi_gpu.create_session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(1, epochs + 1):
            if epoch % anneal_lr_freq == 0:
                learning_rate *= anneal_lr_rate
            np.random.shuffle(x_train)
            gen_losses, disc_losses = [], []
            time_train = -time.time()
            for t in range(iters):
                iter = t + 1
                x_batch = x_train[t * batch_size:(t + 1) * batch_size]
                _, g_loss, d_loss, _ = sess.run(
                    [infer, gen_loss, disc_loss, clip_op],
                    feed_dict={x: x_batch,
                               learning_rate_ph: learning_rate,
                               is_training: True})
                gen_losses.append(g_loss)
                disc_losses.append(d_loss)

                if iter % print_freq == 0:
                    print('Epoch={} Iter={} ({:.3f}s/iter): '
                          'Gen loss = {} Disc loss = {}'.
                          format(epoch, iter,
                                 (time.time() + time_train) / print_freq,
                                 np.mean(gen_losses), np.mean(disc_losses)))
                    gen_losses = []
                    disc_losses = []

                if iter % test_freq == 0:
                    time_test = -time.time()
                    images = sess.run(eval_x_gen,
                                      feed_dict={is_training: False})
                    name = "results/wgan/wgan.epoch.{}.iter.{}.png".format(
                        epoch, iter)
                    save_image_collections(images, name, scale_each=True)
                    time_test += time.time()

                if iter % save_freq == 0:
                    save_path = "results/wgan/wgan.epoch.{}.iter.{}.ckpt". \
                        format(epoch, iter)
                    saver.save(sess, save_path)

                if iter % print_freq == 0:
                    time_train = -time.time()
