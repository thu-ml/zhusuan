#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os

import tensorflow as tf
from six.moves import range, zip
import numpy as np
import zhusuan as zs

from examples import conf
from examples.utils import dataset


@zs.meta_bayesian_net(scope="bnn", reuse_variables=True)
def build_bnn(layer_sizes, n_particles):
    bn = zs.BayesianNet()
    x = bn.input("x")
    h = tf.tile(x[None, ...], [n_particles, 1, 1])
    for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        logstd = tf.get_variable("layer"+str(i)+"logstd", [n_out, n_in + 1],
                                 initializer=tf.zeros_initializer())
        bn.output("layer"+str(i)+"logstd", logstd)
        w = bn.normal("w" + str(i), tf.zeros([n_out, n_in + 1]), logstd=logstd,
                      group_ndims=2, n_samples=n_particles)
        h = tf.concat([h, tf.ones(tf.shape(h)[:-1])[..., None]], -1)
        h = tf.einsum("imk,ijk->ijm", w, h) / tf.sqrt(
            tf.to_float(tf.shape(h)[2]))
        if i < len(layer_sizes) - 2:
            h = tf.nn.relu(h)

    y_mean = bn.output("y_mean", tf.squeeze(h, 2))
    y_logstd = -0.95
    bn.normal("y", y_mean, logstd=y_logstd)
    return bn


def main():
    tf.set_random_seed(1237)
    np.random.seed(2345)

    # Load UCI Boston housing data
    data_path = os.path.join(conf.data_dir, "protein.data")
    x_train, y_train, x_valid, y_valid, x_test, y_test = \
        dataset.load_uci_protein_data(data_path)
    x_train = np.vstack([x_train, x_valid])
    y_train = np.hstack([y_train, y_valid])
    n_train, x_dim = x_train.shape

    # Standardize data
    x_train, x_test, _, _ = dataset.standardize(x_train, x_test)
    y_train, y_test, mean_y_train, std_y_train = dataset.standardize(
        y_train, y_test)

    # Define model parameters
    n_hiddens = [50]

    # Build the computation graph
    n_particles = 20
    x = tf.placeholder(tf.float32, shape=[None, x_dim])
    y = tf.placeholder(tf.float32, shape=[None])
    layer_sizes = [x_dim] + n_hiddens + [1]
    w_names = ["w" + str(i) for i in range(len(layer_sizes) - 1)]
    wv = []
    for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1],
                                      layer_sizes[1:])):
        wv.append(tf.Variable(
            tf.random_uniform([n_particles, n_out, n_in + 1])*4-2))

    meta_model = build_bnn(layer_sizes, n_particles)

    def log_joint(bn):
        log_pws = bn.cond_log_prob(w_names)
        log_py_xw = bn.cond_log_prob('y')
        return tf.add_n(log_pws) + tf.reduce_mean(log_py_xw, 1) * n_train

    meta_model.log_joint = log_joint

    record_full = tf.placeholder(tf.bool, shape=[])

    # sgmcmc = zs.SGLD(learning_rate=1e-5, add_noise=True)
    sgmcmc = zs.SGHMC(learning_rate=2e-6, friction=0.2, n_iter_resample_v=1000,
                      second_order=True)
    # sgmcmc = zs.SGNHT(learning_rate=1e-5, variance_extra=0.1, tune_rate=50.,
    #                   second_order=True)
    latent = dict(zip(w_names, wv))
    sample_op, new_w, sample_info = sgmcmc.sample(meta_model,
                                                  observed={'x': x, 'y': y},
                                                  latent=latent,
                                                  record_full=record_full)

    esti_logstds = [0.5*tf.log(tf.reduce_mean(w*w, axis=0)) for w in wv]
    output_logstds = dict(zip(w_names,
                              [0.5*tf.log(tf.reduce_mean(w*w)) for w in wv]))
    assign_ops = [sgmcmc.bn["layer"+str(i)+"logstd"].assign(logstd)
                  for (i, logstd) in enumerate(esti_logstds)]
    assign_op = tf.group(assign_ops)

    # prediction: rmse & log likelihood
    y_mean = sgmcmc.bn["y_mean"]
    y_pred = tf.reduce_mean(y_mean, 0)

    # Define training/evaluation parameters
    epochs = 500
    batch_size = 100
    iters = (n_train-1) // batch_size + 1

    preds = []
    epochs_ave_pred = 1

    # Run the inference
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(1, epochs + 1):
            perm = np.random.permutation(x_train.shape[0])
            x_train = x_train[perm, :]
            y_train = y_train[perm]
            _, w, info = sess.run([sample_op, new_w, sample_info],
                                  feed_dict={x: x_train,
                                             y: y_train,
                                             record_full: True})
            for t in range(iters):
                x_batch = x_train[t * batch_size:(t + 1) * batch_size]
                y_batch = y_train[t * batch_size:(t + 1) * batch_size]
                _, w, info = sess.run([sample_op, new_w, sample_info],
                                      feed_dict={x: x_batch,
                                                 y: y_batch,
                                                 record_full: False})
            # print(info)
            sess.run(assign_op)

            test_pred = sess.run(y_pred, feed_dict={x: x_test})
            preds.append(test_pred)
            pred = np.mean(preds[-epochs_ave_pred:], axis=0)

            test_rmse = np.sqrt(np.mean((pred - y_test) ** 2)) * std_y_train
            print('>> Epoch {} Test = {} logstds = {}'
                  .format(epoch, test_rmse, sess.run(output_logstds)))

if __name__ == "__main__":
    main()
