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
import zhusuan as zs

from examples import conf
from examples.utils import dataset, multi_gpu, save_image_collections
from examples.utils.multi_gpu import FLAGS


@zs.reuse('generator')
def generator(observed, n, n_z, is_training):
    with zs.BayesianNet(observed=observed) as generator:
        z_min = -tf.ones([n, n_z])
        z_max = tf.ones([n, n_z])
        z = zs.Uniform('z', z_min, z_max)
        lx_z = tf.reshape(z, [-1, 1, 1, n_z])
        ngf = 32
        lx_z = tf.layers.conv2d_transpose(lx_z, ngf * 4, 3, use_bias=False)
        lx_z = tf.layers.batch_normalization(lx_z, training=is_training,
                                             scale=False)
        lx_z = tf.nn.relu(lx_z)
        lx_z = tf.layers.conv2d_transpose(lx_z, ngf * 2, 5, use_bias=False)
        lx_z = tf.layers.batch_normalization(lx_z, training=is_training,
                                             scale=False)
        lx_z = tf.nn.relu(lx_z)
        lx_z = tf.layers.conv2d_transpose(lx_z, ngf, 5, strides=(2, 2),
                                          padding='same', use_bias=False)
        lx_z = tf.layers.batch_normalization(lx_z, training=is_training,
                                             scale=False)
        lx_z = tf.nn.relu(lx_z)
        lx_z = tf.layers.conv2d_transpose(
            lx_z, 1, 5, strides=(2, 2), padding='same', activation=tf.sigmoid)
    return generator, lx_z


@zs.reuse('discriminator')
def discriminator(x, is_training):
    ndf = 16
    lc_x = tf.layers.conv2d(x, ndf, 5, strides=(2, 2),
                            padding='same', use_bias=False)
    lc_x = tf.layers.batch_normalization(lc_x, training=is_training,
                                         scale=False)
    lc_x = tf.nn.relu(lc_x)
    lc_x = tf.layers.conv2d(lc_x, ndf * 2, 5, strides=(2, 2),
                            padding='same', use_bias=False)
    lc_x = tf.layers.batch_normalization(lc_x, training=is_training,
                                         scale=False)
    lc_x = tf.nn.relu(lc_x)
    lc_x = tf.layers.conv2d(lc_x, ndf * 4, 5, use_bias=False)
    lc_x = tf.layers.batch_normalization(lc_x, training=is_training,
                                         scale=False)
    lc_x = tf.nn.relu(lc_x)
    lc_x = tf.reshape(lc_x, [-1, ndf * 4 * 3 * 3])
    critic = tf.layers.dense(lc_x, 1)
    return critic


def main():
    tf.set_random_seed(1234)
    np.random.seed(1234)

    # Load MINST
    data_path = os.path.join(conf.data_dir, 'mnist.pkl.gz')
    x_train, t_train, x_valid, t_valid, x_test, t_test = \
        dataset.load_mnist_realval(data_path)
    n_xl = 28
    n_channels = 1
    x_train = np.vstack([x_train, x_valid]).astype(np.float32).reshape(
        (-1, n_xl, n_xl, n_channels))

    # Define model parameters
    n_z = 40

    # Define training/evaluation parameters
    epochs = 1000
    batch_size = 64 * FLAGS.num_gpus
    gen_size = 100
    iters = x_train.shape[0] // batch_size
    print_freq = 100
    save_freq = 100

    # Build the computation graph
    is_training = tf.placeholder(tf.bool, shape=[], name='is_training')
    x = tf.placeholder(tf.float32, shape=(None, n_xl, n_xl, n_channels),
                       name='x')
    optimizer = tf.train.RMSPropOptimizer(learning_rate=0.0002, decay=0.5)

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
    w_distance = -disc_loss
    grads = multi_gpu.average_gradients(tower_grads)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        infer_op = optimizer.apply_gradients(grads)

    # Clip weights of the critic to ensure 1-Lipschitz
    disc_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                      scope='discriminator')
    with tf.control_dependencies([infer_op]):
        clip_op = tf.group(
            *[var.assign(tf.clip_by_value(var, -0.01, 0.01)) for var
              in disc_var_list])

    # Generate images
    _, eval_x_gen = generator(None, gen_size, n_z, False)

    # Run the inference
    with multi_gpu.create_session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(1, epochs + 1):
            np.random.shuffle(x_train)
            w_losses = []
            time_train = -time.time()
            for t in range(iters):
                iter = t + 1
                x_batch = x_train[t * batch_size:(t + 1) * batch_size]
                _, _, w_loss = sess.run(
                    [infer_op, clip_op, w_distance],
                    feed_dict={x: x_batch, is_training: True})
                w_losses.append(w_loss)

                if iter % print_freq == 0:
                    print('Epoch={} Iter={} ({:.3f}s/iter): '
                          'wasserstein distance = {}'.
                          format(epoch, iter,
                                 (time.time() + time_train) / print_freq,
                                 np.mean(w_losses)))
                    w_losses = []

                if iter % save_freq == 0:
                    images = sess.run(eval_x_gen)
                    name = "results/wgan/wgan.epoch.{}.iter.{}.png".format(
                        epoch, iter)
                    save_image_collections(images, name, scale_each=True)

                if iter % print_freq == 0:
                    time_train = -time.time()


if __name__ == "__main__":
    main()
