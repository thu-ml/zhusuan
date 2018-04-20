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
from examples.utils import save_image_collections, conv2d_transpose


def deconv_resnet_block(input_, out_shape, resize=False):
    if not resize:
        lx_z = conv2d_transpose(input_, out_shape, kernel_size=(3, 3),
                                stride=(1, 1))
        lx_z = conv2d_transpose(lx_z, out_shape, kernel_size=(3, 3),
                                stride=(1, 1), activation_fn=None)
        lx_z += input_
    else:
        lx_z = conv2d_transpose(input_, input_.get_shape().as_list()[1:],
                                kernel_size=(3, 3), stride=(1, 1))
        lx_z = conv2d_transpose(lx_z, out_shape, kernel_size=(3, 3),
                                stride=(2, 2), activation_fn=None)
        residual = conv2d_transpose(input_, out_shape, kernel_size=(3, 3),
                                    stride=(2, 2), activation_fn=None)
        lx_z += residual
    lx_z = tf.nn.relu(lx_z)
    return lx_z


def conv_resnet_block(input_, out_channel, resize=False):
    if not resize:
        lz_x = tf.layers.conv2d(input_, out_channel, 3, padding="same",
                                activation=tf.nn.relu)
        lz_x = tf.layers.conv2d(lz_x, out_channel, 3, padding="same")
        lz_x += input_
    else:
        lz_x = tf.layers.conv2d(input_, out_channel, 3, strides=(2, 2),
                                padding="same", activation=tf.nn.relu)
        lz_x = tf.layers.conv2d(lz_x, out_channel, 3, padding="same")
        residual = tf.layers.conv2d(input_, out_channel, 3, strides=(2, 2),
                                    padding="same")
        lz_x += residual
    lz_x = tf.nn.relu(lz_x)
    return lz_x


@zs.reuse("model")
def vae_conv(observed, n, x_dim, z_dim, n_particles, nf=16):
    with zs.BayesianNet(observed=observed) as model:
        z_mean = tf.zeros([n, z_dim])
        z = zs.Normal("z", z_mean, std=1., group_ndims=1,
                      n_samples=n_particles)
        lx_z = tf.layers.dense(z, 7 * 7 * nf * 2, activation=tf.nn.relu)
        lx_z = tf.reshape(lx_z, [-1, 7, 7, nf * 2])
        lx_z = deconv_resnet_block(lx_z, [7, 7, nf * 2])
        lx_z = deconv_resnet_block(lx_z, [14, 14, nf * 2], resize=True)
        lx_z = deconv_resnet_block(lx_z, [14, 14, nf * 2])
        lx_z = deconv_resnet_block(lx_z, [28, 28, nf], resize=True)
        lx_z = deconv_resnet_block(lx_z, [28, 28, nf])
        lx_z = conv2d_transpose(lx_z, [28, 28, 1], kernel_size=(3, 3),
                                stride=(1, 1), activation_fn=None)
        x_logits = tf.reshape(lx_z, [n_particles, -1, x_dim])
        x = zs.Bernoulli("x", x_logits, group_ndims=1)
    return model, x_logits


@zs.reuse("variational")
def q_net(x, z_dim, n_particles, nf=16):
    with zs.BayesianNet() as variational:
        lz_x = 2 * tf.to_float(x) - 1
        lz_x = tf.reshape(lz_x, [-1, 28, 28, 1])
        lz_x = tf.layers.conv2d(lz_x, nf, 3, padding="same",
                                activation=tf.nn.relu)
        lz_x = conv_resnet_block(lz_x, nf)
        lz_x = conv_resnet_block(lz_x, nf * 2, resize=True)
        lz_x = conv_resnet_block(lz_x, nf * 2)
        lz_x = conv_resnet_block(lz_x, nf * 2, resize=True)
        lz_x = conv_resnet_block(lz_x, nf * 2)
        lz_x = tf.layers.flatten(lz_x)
        lz_x = tf.layers.dense(lz_x, 500, activation=tf.nn.relu)
        z_mean = tf.layers.dense(lz_x, z_dim)
        z_logstd = tf.layers.dense(lz_x, z_dim)
        z = zs.Normal("z", z_mean, logstd=z_logstd, group_ndims=1,
                      n_samples=n_particles)
    return variational


def main():
    tf.set_random_seed(1234)
    np.random.seed(1234)

    # Load MNIST
    data_path = os.path.join(conf.data_dir, "mnist.pkl.gz")
    x_train, t_train, x_valid, t_valid, x_test, t_test = \
        dataset.load_mnist_realval(data_path)
    x_train = np.vstack([x_train, x_valid])
    x_test = np.random.binomial(1, x_test, size=x_test.shape)
    x_dim = x_train.shape[1]

    # Define model parameters
    z_dim = 32

    # Build the computation graph
    n_particles = tf.placeholder(tf.int32, shape=[], name="n_particles")
    x_input = tf.placeholder(tf.float32, shape=[None, x_dim])
    x = tf.to_int32(tf.random_uniform(tf.shape(x_input)) <= x_input)
    n = tf.shape(x)[0]

    def log_joint(observed):
        model, _ = vae_conv(observed, n, x_dim, z_dim, n_particles)
        log_pz, log_px_z = model.local_log_prob(["z", "x"])
        return log_pz + log_px_z

    variational = q_net(x, z_dim, n_particles)
    qz_samples, log_qz = variational.query("z", outputs=True,
                                           local_log_prob=True)
    lower_bound = zs.variational.elbo(log_joint,
                                      observed={"x": x},
                                      latent={"z": [qz_samples, log_qz]},
                                      axis=0)
    cost = tf.reduce_mean(lower_bound.sgvb())
    lower_bound = tf.reduce_mean(lower_bound)

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5)
    infer_op = optimizer.minimize(cost)

    # Generate images
    n_gen = 100
    _, x_logits = vae_conv({}, n_gen, x_dim, z_dim, 1)
    x_gen = tf.reshape(tf.sigmoid(x_logits), [-1, 28, 28, 1])

    # Define training/evaluation parameters
    epochs = 3000
    batch_size = 128
    iters = x_train.shape[0] // batch_size
    save_freq = 10
    test_freq = 10
    test_batch_size = 400
    test_iters = x_test.shape[0] // test_batch_size
    result_path = "results/vae_conv"

    # Run the inference
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(1, epochs + 1):
            time_epoch = -time.time()
            np.random.shuffle(x_train)
            lbs = []
            for t in range(iters):
                x_batch = x_train[t * batch_size:(t + 1) * batch_size]
                _, lb = sess.run([infer_op, lower_bound],
                                 feed_dict={x_input: x_batch,
                                            n_particles: 1})
                lbs.append(lb)
            time_epoch += time.time()
            print("Epoch {} ({:.1f}s): Lower bound = {}".format(
                epoch, time_epoch, np.mean(lbs)))

            if epoch % test_freq == 0:
                time_test = -time.time()
                test_lbs = []
                for t in range(test_iters):
                    test_x_batch = x_test[
                        t * test_batch_size: (t + 1) * test_batch_size]
                    test_lb = sess.run(lower_bound,
                                       feed_dict={x: test_x_batch,
                                                  n_particles: 1})
                    test_lbs.append(test_lb)
                time_test += time.time()
                print(">>> TEST ({:.1f}s)".format(time_test))
                print(">> Test lower bound = {}".format(np.mean(test_lbs)))

            if epoch % save_freq == 0:
                print("Saving images...")
                images = sess.run(x_gen)
                name = os.path.join(result_path,
                                    "vae.epoch.{}.png".format(epoch))
                save_image_collections(images, name)


if __name__ == "__main__":
    main()
