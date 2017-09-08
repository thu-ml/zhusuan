#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import tensorflow as tf
import numpy as np
import zhusuan as zs

from examples import conf
from examples.utils import dataset


def load_movielens1m_mapped(path):
    num_movies, num_users, train, valid, test = load_movielens1m(path)

    user_movie = []
    user_movie_score = []
    for i in range(num_users):
        user_movie.append([])
        user_movie_score.append([])
    movie_user = []
    movie_user_score = []
    for i in range(num_users):
        movie_user.append([])
        movie_user_score.append([])

    for i in range(np.shape(train)[0]):
        user_id = train[i, 0]
        movie_id = train[i, 1]
        rating = train[i, 2]
        user_movie[user_id].append(movie_id)
        user_movie_score[user_id].append(rating)
        movie_user[movie_id].append(user_id)
        movie_user_score[movie_id].append(rating)

    return num_movies, num_users, train, valid, test, \
        user_movie, user_movie_score, movie_user, movie_user_score


def select_from_axis1(para, indices):
    gather_para = tf.transpose(para, perm=[1, 0, 2])
    gather_para = tf.gather(gather_para, indices)
    gather_para = tf.transpose(gather_para, perm=[1, 0, 2])
    return gather_para


def pmf(observed, n, m, D, n_particles, select_u, select_v,
        alpha_u, alpha_v, alpha_pred):
    with zs.BayesianNet(observed=observed) as model:
        mu_u = tf.zeros(shape=[n, D])
        log_std_u = tf.ones(shape=[n, D]) * tf.log(alpha_u)
        u = zs.Normal('u', mu_u, log_std_u,
                      n_samples=n_particles, group_event_ndims=1)
        mu_v = tf.zeros(shape=[m, D])
        log_std_v = tf.ones(shape=[m, D]) * tf.log(alpha_v)
        v = zs.Normal('v', mu_v, log_std_v,
                      n_samples=n_particles, group_event_ndims=1)
        gather_u = select_from_axis1(u, select_u)  # [K, batch, D]
        gather_v = select_from_axis1(v, select_v)  # [K, batch, D]
        pred_mu = tf.reduce_sum(gather_u * gather_v, axis=2)
        r = zs.Normal('r', tf.sigmoid(pred_mu), tf.log(alpha_pred))
    return model, tf.sigmoid(pred_mu)


def select_from_corpus(l, r, u_v, u_v_score):
    sv = []
    tr = []
    map_uid_idx = {}
    for i in range(r - l):
        # consider the no-rating film / no-rating people
        try:
            sv = sv + u_v[l + i]
            tr = tr + u_v_score[l + i]
        except:
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
        except:
            pass
    return nv, np.array(sv), tr, ssu, ssv


if __name__ == '__main__':
    np.random.seed(1234)
    tf.set_random_seed(1237)
    M, N, train_data, valid_data, test_data, user_movie, \
        user_movie_score, movie_user, movie_user_score \
        = load_movielens1m_mapped('data/ml-1m.zip')
    old_M = M
    old_N = N

    # set configurations and hyper parameters
    N_train = np.shape(train_data)[0]
    N_test = np.shape(test_data)[0]
    N_valid = np.shape(valid_data)[0]
    D = 30
    batch_size = 100000
    test_batch_size = 100000
    valid_batch_size = 100000
    learning_rate = 0.005
    K = 8
    num_steps = 500
    test_freq = 10
    valid_freq = 10
    train_iters = (N_train + batch_size - 1) // batch_size
    valid_iters = (N_valid + valid_batch_size - 1) // valid_batch_size
    test_iters = (N_test + test_batch_size - 1) // test_batch_size

    # paralleled
    chunk_size = 50
    N = (N + chunk_size - 1) / chunk_size
    N *= chunk_size
    M = (M + chunk_size - 1) / chunk_size
    M *= chunk_size

    # Selection
    select_u = tf.placeholder(tf.int32, shape=[None, ], name='s_u')
    select_v = tf.placeholder(tf.int32, shape=[None, ], name='s_v')
    subselect_u = tf.placeholder(tf.int32, shape=[None, ], name='ss_u')
    subselect_v = tf.placeholder(tf.int32, shape=[None, ], name='ss_v')
    alpha_u = 1.0
    alpha_v = 1.0
    alpha_pred = 0.2 / 4.0

    # Find non-trained files or peoples
    trained_movie = [False] * old_M
    trained_user = [False] * old_N
    for i in range(N_train):
        trained_user[train_data[i, 0]] = True
        trained_movie[train_data[i, 1]] = True
    us = 0
    vs = 0
    for i in range(old_N):
        us += trained_user[i]
    for j in range(old_M):
        vs += trained_movie[j]
    print('Untrained users = %d, untrained movied = %d'
          % (old_N - us, old_M - vs))
    trained_movie = tf.constant(trained_movie, dtype=tf.bool)
    trained_user = tf.constant(trained_user, dtype=tf.bool)

    # Define samples as variables
    Us = []
    Vs = []
    for i in range(N / chunk_size):
        ui = tf.get_variable('u_chunk_%d' % i, shape=[K, chunk_size, D],
                             initializer=tf.random_normal_initializer(0, 0.1),
                             trainable=False)
        Us.append(ui)
    for i in range(M / chunk_size):
        vi = tf.get_variable('v_chunk_%d' % i, shape=[K, chunk_size, D],
                             initializer=tf.random_normal_initializer(0, 0.1),
                             trainable=False)
        Vs.append(vi)
    U = tf.concat(Us, axis=1)
    V = tf.concat(Vs, axis=1)

    # Define models for prediction
    true_rating = tf.placeholder(tf.float32, shape=[None, ],
                                 name='true_rating')
    normalized_rating = (true_rating - 1.0) / 4.0
    true_rating_2d = tf.tile(tf.expand_dims(normalized_rating, 0), [K, 1])
    _, pred_rating = pmf({'u': U, 'v': V, 'r': true_rating_2d}, N, M, D, K,
                         select_u, select_v, alpha_u, alpha_v, alpha_pred)
    pred_rating = tf.reduce_mean(pred_rating, axis=0)
    error = pred_rating - normalized_rating
    rmse = tf.sqrt(tf.reduce_mean(error * error)) * 4

    # Define models for HMC
    n = tf.placeholder(tf.int32, shape=[], name='n')
    m = tf.placeholder(tf.int32, shape=[], name='m')

    def log_joint(observed):
        model, _ = pmf(observed, n, m, D, K, subselect_u,
                       subselect_v, alpha_u, alpha_v, alpha_pred)
        log_pu, log_pv = model.local_log_prob(['u', 'v'])    # [K, N], [K, M]
        log_pr = model.local_log_prob('r')                   # [K, batch]
        log_pu = tf.reduce_sum(log_pu, axis=1)
        log_pv = tf.reduce_sum(log_pv, axis=1)
        log_pr = tf.reduce_sum(log_pr, axis=1)
        return log_pu + log_pv + log_pr

    hmc_u = zs.HMC(step_size=1e-3, n_leapfrogs=10, adapt_step_size=None,
                   target_acceptance_rate=0.9)
    hmc_v = zs.HMC(step_size=1e-3, n_leapfrogs=10, adapt_step_size=None,
                   target_acceptance_rate=0.9)
    target_u = select_from_axis1(U, select_u)
    target_v = select_from_axis1(V, select_v)

    candidate_sample_u = \
        tf.get_variable('cand_sample_chunk_u', shape=[K, chunk_size, D],
                        initializer=tf.random_normal_initializer(0, 0.1),
                        trainable=True)
    candidate_sample_v = \
        tf.get_variable('cand_sample_chunk_v', shape=[K, chunk_size, D],
                        initializer=tf.random_normal_initializer(0, 0.1),
                        trainable=True)
    sample_u_op, sample_u_info = \
        hmc_u.sample(log_joint, {'r': true_rating_2d, 'v': target_v},
                                {'u': candidate_sample_u})
    sample_v_op, sample_v_info = \
        hmc_v.sample(log_joint, {'r': true_rating_2d, 'u': target_u},
                                {'v': candidate_sample_v})

    candidate_idx_u = tf.placeholder(tf.int32, shape=[chunk_size, ],
                                     name='cand_u_chunk')
    candidate_idx_v = tf.placeholder(tf.int32, shape=[chunk_size, ],
                                     name='cand_v_chunk')
    candidate_u = select_from_axis1(U, candidate_idx_u)  # [K, chunk_size, D]
    candidate_v = select_from_axis1(V, candidate_idx_v)  # [K, chunk_size, D]

    trans_cand_U = tf.assign(candidate_sample_u, candidate_u)
    trans_cand_V = tf.assign(candidate_sample_v, candidate_v)

    trans_us_cand = []
    for i in range(N / chunk_size):
        trans_us_cand.append(tf.assign(Us[i], candidate_sample_u))
    trans_vs_cand = []
    for i in range(M / chunk_size):
        trans_vs_cand.append(tf.assign(Vs[i], candidate_sample_v))

    params = tf.trainable_variables()
    for i in params:
        print(i.name, i.get_shape())

    saver = tf.train.Saver(max_to_keep=10)

    # Run the inference
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for step in range(1, num_steps + 1):
            epoch_time = -time.time()
            for i in range(N / chunk_size):
                nv, sv, tr, ssu, ssv = select_from_corpus(i * chunk_size,
                                                          (i + 1) * chunk_size,
                                                          user_movie,
                                                          user_movie_score)
                _ = sess.run(trans_cand_U,
                             feed_dict={
                                 candidate_idx_u: range(i * chunk_size,
                                                        (i + 1) * chunk_size)})
                _ = sess.run(sample_u_op, feed_dict={select_v: sv,
                                                     true_rating: tr,
                                                     subselect_u: ssu,
                                                     subselect_v: ssv,
                                                     n: chunk_size,
                                                     m: nv})
                _ = sess.run(trans_us_cand[i])
            for i in range(M / chunk_size):
                nu, su, tr, ssv, ssu = select_from_corpus(i * chunk_size,
                                                          (i + 1) * chunk_size,
                                                          movie_user,
                                                          movie_user_score)
                _ = sess.run(trans_cand_V,
                             feed_dict={
                                 candidate_idx_v: range(i * chunk_size,
                                                        (i + 1) * chunk_size)})
                _ = sess.run(sample_v_op, feed_dict={select_u: su,
                                                     true_rating: tr,
                                                     subselect_u: ssu,
                                                     subselect_v: ssv,
                                                     n: nu,
                                                     m: chunk_size})
                _ = sess.run(trans_vs_cand[i])
            epoch_time += time.time()

            train_rmse = []
            time_train = -time.time()
            for t in range(train_iters):
                ed_pos = min((t + 1) * batch_size, N_train + 1)
                n_batch = ed_pos - t * batch_size
                su = train_data[t * batch_size:ed_pos, 0]
                sv = train_data[t * batch_size:ed_pos, 1]
                tr = train_data[t * batch_size:ed_pos, 2]
                re = sess.run(rmse, feed_dict={select_u: su, select_v: sv,
                                               true_rating: tr})
                train_rmse.append(re)
            time_train += time.time()

            x = np.mean(train_rmse)

            print('Step {}({:.1f}s): Finished!, rmse ({:.1f}s) = {}'.format
                  (step, epoch_time, time_train, np.mean(train_rmse)))

            if step % test_freq == 0:
                test_rmse = []
                time_test = -time.time()
                for t in range(test_iters):
                    ed_pos = min((t + 1) * test_batch_size, N_test + 1)
                    n_batch = ed_pos - t * test_batch_size
                    su = test_data[t * test_batch_size:ed_pos, 0]
                    sv = test_data[t * test_batch_size:ed_pos, 1]
                    tr = test_data[t * test_batch_size:ed_pos, 2]
                    re = sess.run(rmse, feed_dict={select_u: su, select_v: sv,
                                                   true_rating: tr})
                    test_rmse.append(re)
                time_test += time.time()
                print('>>> VALIDATION ({:.1f}s)'.format(time_test))
                print('>> Validation rmse = {}'.format((np.mean(test_rmse))))

            if step % valid_freq == 0:
                valid_rmse = []
                time_valid = -time.time()
                for t in range(valid_iters):
                    ed_pos = min((t + 1) * valid_batch_size, N_test + 1)
                    n_batch = ed_pos - t * valid_batch_size
                    su = valid_data[t * valid_batch_size:ed_pos, 0]
                    sv = valid_data[t * valid_batch_size:ed_pos, 1]
                    tr = valid_data[t * valid_batch_size:ed_pos, 2]
                    re = sess.run(rmse, feed_dict={select_u: su, select_v: sv,
                                                   true_rating: tr})
                    valid_rmse.append(re)
                time_valid += time.time()
                print('>>> TEST ({:.1f}s)'.format(time_valid))
                print('>> Test rmse = {}'.format((np.mean(valid_rmse))))

