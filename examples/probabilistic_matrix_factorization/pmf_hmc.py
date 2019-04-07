#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import time

from six.moves import range
import tensorflow as tf
import numpy as np
import zhusuan as zs

from examples import conf
from examples.utils import dataset, average_rmse_over_batches


@zs.meta_bayesian_net(scope="pmf", reuse_variables=True)
def pmf(n, m, D, n_particles, select_u, select_v, alpha_u, alpha_v, alpha_pred):
    bn = zs.BayesianNet()
    mu_u = tf.zeros(shape=[n, D])
    u = bn.normal("u", mu_u, std=alpha_u, n_samples=n_particles, group_ndims=1)
    mu_v = tf.zeros(shape=[m, D])
    v = bn.normal("v", mu_v, std=alpha_v, n_samples=n_particles, group_ndims=1)
    gather_u = tf.gather(u, select_u, axis=1)  # [K, batch, D]
    gather_v = tf.gather(v, select_v, axis=1)  # [K, batch, D]
    r_logits = tf.reduce_sum(gather_u * gather_v, axis=2)
    bn.deterministic("r_pred", tf.sigmoid(r_logits))
    bn.normal("r", tf.sigmoid(r_logits), std=alpha_pred)
    return bn


def select_from_corpus(l, r, u_v, u_v_score):
    sv = []
    tr = []
    map_uid_idx = {}
    for i in range(r - l):
        # consider the no-rating film / no-rating people
        try:
            sv = sv + u_v[l + i]
            tr = tr + u_v_score[l + i]
        except Exception:
            pass
    sv = list(set(sv))
    nv = len(sv)
    for i in range(nv):
        map_uid_idx[sv[i]] = i
    ssu = []
    ssv = []
    for i in range(r - l):
        # consider the no-rating film / no-rating people
        try:
            lt = u_v[l + i]
            ssu += [i] * len(lt)
            for j in range(len(lt)):
                ssv.append(map_uid_idx[lt[j]])
        except Exception:
            pass
    return nv, np.array(sv), tr, ssu, ssv


def main():
    np.random.seed(1234)
    tf.set_random_seed(1237)
    M, N, train_data, valid_data, test_data, user_movie, user_movie_score, \
        movie_user, movie_user_score = dataset.load_movielens1m_mapped(
            os.path.join(conf.data_dir, 'ml-1m.zip'))

    # set configurations and hyper parameters
    D = 30
    batch_size = 100000
    K = 8
    n_epochs = 500
    eval_freq = 10

    # paralleled
    chunk_size = 50
    N = (N + chunk_size - 1) // chunk_size
    N *= chunk_size
    M = (M + chunk_size - 1) // chunk_size
    M *= chunk_size

    # Selection
    neighbor_u = tf.placeholder(tf.int32, shape=[None], name="neighbor_u")
    neighbor_v = tf.placeholder(tf.int32, shape=[None], name="neighbor_v")
    select_u = tf.placeholder(tf.int32, shape=[None], name="select_u")
    select_v = tf.placeholder(tf.int32, shape=[None], name="select_v")
    alpha_u = 1.0
    alpha_v = 1.0
    alpha_pred = 0.2 / 4.0

    # Define samples as variables
    Us = []
    Vs = []
    for i in range(N // chunk_size):
        ui = tf.get_variable('u_chunk_%d' % i, shape=[K, chunk_size, D],
                             initializer=tf.random_normal_initializer(0, 0.1),
                             trainable=False)
        Us.append(ui)
    for i in range(M // chunk_size):
        vi = tf.get_variable('v_chunk_%d' % i, shape=[K, chunk_size, D],
                             initializer=tf.random_normal_initializer(0, 0.1),
                             trainable=False)
        Vs.append(vi)
    U = tf.concat(Us, axis=1)
    V = tf.concat(Vs, axis=1)

    n = tf.placeholder(tf.int32, shape=[], name='n')
    m = tf.placeholder(tf.int32, shape=[], name='m')
    model = pmf(n, m, D, K, select_u, select_v, alpha_u, alpha_v,
                     alpha_pred)

    # prediction
    true_rating = tf.placeholder(tf.float32, shape=[None], name='true_rating')
    normalized_rating = (true_rating - 1.0) / 4.0
    pred_rating = model.observe(u=U, v=V)["r"]
    pred_rating = tf.reduce_mean(pred_rating, axis=0)
    rmse = tf.sqrt(
        tf.reduce_mean(tf.square(pred_rating - normalized_rating))) * 4

    hmc_u = zs.HMC(step_size=1e-3, n_leapfrogs=10, adapt_step_size=None,
                   target_acceptance_rate=0.9)
    hmc_v = zs.HMC(step_size=1e-3, n_leapfrogs=10, adapt_step_size=None,
                   target_acceptance_rate=0.9)
    target_u = tf.gather(U, neighbor_u, axis=1)
    target_v = tf.gather(V, neighbor_v, axis=1)

    candidate_sample_u = tf.get_variable(
        'cand_sample_chunk_u', shape=[K, chunk_size, D],
        initializer=tf.random_normal_initializer(0, 0.1), trainable=True)
    candidate_sample_v = tf.get_variable(
        'cand_sample_chunk_v', shape=[K, chunk_size, D],
        initializer=tf.random_normal_initializer(0, 0.1), trainable=True)

    def log_joint(bn):
        log_pu, log_pv = bn.cond_log_prob(['u', 'v'])    # [K, N], [K, M]
        log_pr = bn.cond_log_prob('r')                   # [K, batch]
        log_pu = tf.reduce_sum(log_pu, axis=-1)
        log_pv = tf.reduce_sum(log_pv, axis=-1)
        log_pr = tf.reduce_sum(log_pr, axis=-1)
        return log_pu + log_pv + log_pr

    model.log_joint = log_joint

    sample_u_op, sample_u_info = hmc_u.sample(
        model,
        {"r": normalized_rating, "v": target_v},
        {"u": candidate_sample_u})
    sample_v_op, sample_v_info = hmc_v.sample(
        model,
        {"r": normalized_rating, "u": target_u},
        {"v": candidate_sample_v})

    candidate_idx_u = tf.placeholder(tf.int32, shape=[chunk_size],
                                     name='cand_u_chunk')
    candidate_idx_v = tf.placeholder(tf.int32, shape=[chunk_size],
                                     name='cand_v_chunk')
    candidate_u = tf.gather(U, candidate_idx_u, axis=1)  # [K, chunk_size, D]
    candidate_v = tf.gather(V, candidate_idx_v, axis=1)  # [K, chunk_size, D]

    trans_cand_U = tf.assign(candidate_sample_u, candidate_u)
    trans_cand_V = tf.assign(candidate_sample_v, candidate_v)

    trans_us_cand = []
    for i in range(N // chunk_size):
        trans_us_cand.append(tf.assign(Us[i], candidate_sample_u))
    trans_vs_cand = []
    for i in range(M // chunk_size):
        trans_vs_cand.append(tf.assign(Vs[i], candidate_sample_v))

    # Run the inference
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(1, n_epochs + 1):
            epoch_time = -time.time()
            for i in range(N // chunk_size):
                nv, sv, tr, ssu, ssv = select_from_corpus(i * chunk_size,
                                                          (i + 1) * chunk_size,
                                                          user_movie,
                                                          user_movie_score)
                _ = sess.run(trans_cand_U,
                             feed_dict={
                                 candidate_idx_u: list(range(
                                     i * chunk_size, (i + 1) * chunk_size))})
                _ = sess.run(sample_u_op, feed_dict={neighbor_v: sv,
                                                     true_rating: tr,
                                                     select_u: ssu,
                                                     select_v: ssv,
                                                     n: chunk_size,
                                                     m: nv})
                _ = sess.run(trans_us_cand[i])
            for i in range(M // chunk_size):
                nu, su, tr, ssv, ssu = select_from_corpus(i * chunk_size,
                                                          (i + 1) * chunk_size,
                                                          movie_user,
                                                          movie_user_score)
                _ = sess.run(trans_cand_V,
                             feed_dict={
                                 candidate_idx_v: list(range(
                                     i * chunk_size, (i + 1) * chunk_size))})
                _ = sess.run(sample_v_op, feed_dict={neighbor_u: su,
                                                     true_rating: tr,
                                                     select_u: ssu,
                                                     select_v: ssv,
                                                     n: nu,
                                                     m: chunk_size})
                _ = sess.run(trans_vs_cand[i])
            epoch_time += time.time()
            print("Epoch {}: {:.1f}s".format(epoch, epoch_time))

            def _eval(phase, data, batch_size):
                rmses = []
                sizes = []
                time_eval = -time.time()
                n_iters = (data.shape[0] + batch_size - 1) // batch_size
                for t in range(n_iters):
                    su = data[t * batch_size:(t + 1) * batch_size, 0]
                    sv = data[t * batch_size:(t + 1) * batch_size, 1]
                    tr = data[t * batch_size:(t + 1) * batch_size, 2]
                    re = sess.run(rmse, feed_dict={select_u: su,
                                                   select_v: sv,
                                                   n: N,
                                                   m: M,
                                                   true_rating: tr})
                    rmses.append(re)
                    sizes.append(tr.shape[0])
                time_eval += time.time()
                print('>>> {} ({:.1f}s): rmse = {}'.format(
                    phase, time_eval, average_rmse_over_batches(rmses, sizes)))

            _eval("Train", train_data, batch_size)

            if epoch % eval_freq == 0:
                _eval("Validation", valid_data, batch_size)
                _eval("Test", test_data, batch_size)


if __name__ == "__main__":
    main()
