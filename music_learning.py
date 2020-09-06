#!/usr/bin/python3
'''Bayesian Search of transition matrix's weights to map embedding's space to musicVAE space.
In this module the bayesian optimisation process was implemented.
The task of matching embeddings to musicVAE latent space have several difficulties.
To start with, we want to distinguishably compare melodies not in the latent space,
but in the final midi form.

That implies that we can't calculate gradients with respect to weights, so we can't use
family of Gradient Descent Optimisation's algorithms. That's why we can either randomly pick
weights until good metric score or use Bayesian Optimisation, which take into account previous
attempts. That helps to converge faster that Random Search in general.
That means we need to somehow compare different series with not equal lengths. Existing approaches
either calculate area between melodies interpolated to the same length or use some versions of
Dynamic Time Warping algorithm. This module uses it's own midi sequences similarity function.
This function takes into account difference in pitches, difference in durations and
difference in first notes and use Manhattan Distance to get final score. Manhattan distance
is often used in practice to compare signals and melodies sequences are some kind of signals.

Frankly speaking, using bayesian search to find weights of transition matrix is a bad idea.
There are latent_space_size times embedding_size number of parameters to optimize.
Or 512x9~=5k parameters. A rule of thumb is to use 12-16 times iterations as number of parameters.
Considering that bayesian optimisation is O(n^3) due to necessity of finding inverse matrix,
where n is the number of parameters, this optimisation can take years.
To leverage this problem this module use parameters reduction through Gaussian random projection.
So instead of finding matrix weights we search 100 coefficients of linear combination
of randomly generated matrices. Thus, number of parameters was dropped fiftyfold.

It's worth to mention that musciVAE latent space has some garbage dimensions which
affects the generated melodies by little. In the research we have shown that only
10-20 dimensions of musicVAE latent space are vital to melodies generations.

Examples:
    To use this module, you can simply run the following command:
        # python music_learning.py

Todo:
    * Add command line argumets so user can define paths to datasets and musicvae models

.. _Bayesian Search of transition matrix's weights to map embedding's space to musicVAE space:
   https://github.com/Defasium/bayesVec2Midi
'''


from collections import namedtuple
import copy as ccopy
import os
import sys
import math
import time
import re

import pandas as pd
from magenta.models.music_vae import TrainedModel, configs
from magenta import music
from tqdm import tqdm
import joblib
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_curve, roc_auc_score
import GPyOpt
import matplotlib.pyplot as plt

np.random.seed(0)

features = dict(ticks=re.compile(r'ticks_per_quarter: (.+)'),
                qpm=re.compile(r'qpm: (.+)\n'),
                pitch=re.compile(r'pitch: (.+)\n'),
                start_time=re.compile(r'start_time: (.+)\n'),
                end_time=re.compile(r'end_time: (.+)\n'))


def get_batch_random(batch_size, s='train'):
    '''Create batch of APN triplets with a complete random strategy

            Args:
                batch_size (integer)
                s (str, default is 'train'): if s equals 'train' then use 'dataset_train' variable;
                    if s equals 'train_aug' then use 'dataset_aug' global variable;
                    else use 'dataset_test' global variable.
                return_indexes (bool, default is False): if True then instead of triplets with
                images will return triplets with corresponding indexes of that images in dataset.

            Returns:
                triplets (list): list containing 3 tensors A,P,N of shape (batch_size, w, h, c).

    '''
    if s == 'train':
        X = dataset_train
    else:
        X = dataset_test

    _, h = X[0].shape
    # print(X[0].shape)
    # initialize result
    triplets = [np.zeros((batch_size, h)) for i in range(3)]
    a_classes = np.random.randint(0, nb_classes, size=batch_size)
    n_offsets = np.random.randint(1, nb_classes, size=batch_size)
    for i in range(batch_size):
        # Pick one random class for anchor
        anchor_class = a_classes[i]

        # Pick two different random pics for this class => A and P
        [idx_A, idx_P] = np.random.choice(X[anchor_class].shape[0], size=2, replace=False)

        # Pick another class for N, different from anchor_class
        negative_class = (anchor_class + n_offsets[i]) % nb_classes

        # Pick a random pic for this negative class => N
        idx_N = np.random.randint(0, X[negative_class].shape[0])
        # print(idx_A.shape, X[anchor_class][idx_A].shape)
        triplets[0][i] = X[anchor_class][idx_A]
        triplets[1][i] = X[anchor_class][idx_P]
        triplets[2][i] = X[negative_class][idx_N]
    return triplets


def compute_dist(a, b):
    '''Compute l2 distances between embeddings.
        Args:
            a (numpy.ndarray): first embedding of shape (n,).
            b (numpy.ndarray): second embeddings of shape (n,).

        Returns:
            distance (float): l2 distance.

    '''
    return np.sum(np.square(a - b))


def compute_probs(pd1, X, Y):
    '''Compute probabilities of being the right decision
        Args:
            pd1 (pandas.DataFrame): dataset with music embeddings.
            X (numpy.ndarray): tensor of shape (m,h) containing color embeddings to evaluate.
            Y (numpy.ndarray): tensor of shape (m,) containing true class.

        Returns:
            probs : array of shape (m,m) containing distances.

    '''
    m = X.shape[0]
    nbevaluation = int(m * (m - 1) / 2)
    probs = np.zeros((nbevaluation))
    y = np.zeros((nbevaluation))

    embeddings = pd1[['fst', 'p', 'd']].values.tolist()

    # For each pics of our dataset
    k = 0
    for i in range(m):
        # Against all other images
        for j in range(i + 1, m):
            # compute the probability of being the right decision : it should be 1 for right class,
            # 0 for all other classes
            probs[k] = -justify(*embeddings[i], *embeddings[j], penalty_enabled=False)
            if Y[i] == Y[j]:
                y[k] = 1
                # print('{3}:{0} vs {1} : {2}\tSAME'.format(i,j,probs[k],k))
            else:
                y[k] = 0
                # print('{3}:{0} vs {1} : \t\t\t{2}\tDIFF'.format(i,j,probs[k],k))
            k += 1
    return probs, y


def compute_metrics(probs, yprobs):
    '''Compute ROC-AUC score
        Args:
            probs : computed probabilities of class being the right decision.
            yprobs : true values of classes for probabilities.

        Returns:
            fpr : Increasing false positive rates such that element i is the false positive
                rate of predictions with score >= thresholds[i].
            tpr : Increasing true positive rates such that element i is the true positive
                rate of predictions with score >= thresholds[i].
            thresholds : Decreasing thresholds on the decision function used to compute fpr and tpr
                thresholds[0] represents no instances being predicted and
                is arbitrarily set to max(y_score) + 1.
            auc : Area Under the ROC Curve metric.

    '''
    # calculate AUC
    auc = roc_auc_score(yprobs, probs)
    # calculate roc curve
    fpr, tpr, thresholds = roc_curve(yprobs, probs)

    return fpr, tpr, thresholds, auc


def compute_interdist(pd1):
    '''Computes sum of distances between all classes embeddings on our reference test image:
        d(0,1) + d(0,2) + ... + d(0,9) + d(1,2) + d(1,3) + ... d(8,9)
        A good model should have a large distance between all theses embeddings

        Args:
            pd1 (pandas.DataFrame): dataset with music embeddings.

        Returns:
            array of shape (nb_classes, nb_classes)

    '''
    res = np.zeros((9, 9))

    ref_images = []  # np.zeros((nb_classes, 4))

    # generates embeddings for reference images
    for i in range(9):
        ref_images.append(pd1.loc[pd1.color == i, ['fst', 'p', 'd']].values.tolist()[0])
    ref_embeddings = ref_images

    for i in range(9):
        for j in range(9):
            res[i, j] = justify(*ref_embeddings[i], *ref_embeddings[j])
    return res


def draw_interdist(network, n_iteration, path):
    '''Plots interdistance between classes on n_iteration:
        A good model should have a large distance between all theses embeddings

        Args:
            network: Neural network to train outputing embeddings.
            n_iteration: Current number of iterations.

        Returns:
            None.

    '''
    plt.figure(figsize=(7, 7))
    interdist = compute_interdist(network)

    dat = []
    for i in range(nb_classes):
        dat.append(np.delete(interdist[i, :], [i]))

    _, ax = plt.subplots()
    title = 'Evaluating embeddings distance from ' + \
            'each other after {0} iterations'.format(n_iteration)
    ax.set_title(title)
    ax.set_ylim([0, 3])
    plt.xlabel('Classes')
    plt.ylabel('Distance')
    ax.boxplot(dat, showfliers=False, showbox=True)
    locs, _ = plt.xticks()
    plt.xticks(locs, np.arange(nb_classes))
    plt.savefig(os.path.join(path, 'interdist.png'))


def find_nearest(array, value):
    '''Find nearest array element to given value.
            Args:
                array: Array with shape (m, n) or (n, m).
                value: Value with shape (1, n) or (n,).

            Returns:
                array_element with shape (1, n) or (n,).

    '''
    idx = np.searchsorted(array, value, side='left')
    if idx > 0 and (idx == len(array) or
                            math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])):
        return array[idx - 1], idx - 1
    return array[idx], idx


def draw_roc(auc, fpr, tpr, thresholds, path):
    '''Plot ROC-AUC curve.

            Args:
                fpr: false positive rates.
                tpr: true positive rates.
                thresholds: thresholds for auc.
                auc: auc score

            Returns:
                None.

    '''
    # find threshold
    plt.figure(figsize=(7, 7))
    targetfpr = 1e-3
    _, idx = find_nearest(fpr, targetfpr)
    threshold = abs(thresholds[idx])
    recall = tpr[idx]

    # plot no skill
    plt.plot([0, 1], [0, 1], linestyle='--')
    # plot the roc curve for the model
    plt.plot(fpr, tpr, marker='.')
    title = 'AUC: {0:.3f}\nSensitivity : {2:.1%} @FPR={1:.0e}\nThreshold={3})'.format(auc,
                                                                                      targetfpr,
                                                                                      recall,
                                                                                      threshold)
    plt.title(title)
    plt.savefig(os.path.join(path, 'auc.png'))


def get_features_from_midi(s:str):
    '''Convert magenta midi file to three melody components:
        1st note, relative pitches and durations.

            Args:
                s (str): string representation magenta midi object.

            Returns:
                tuple of (int, numpy.ndarray, numpy.ndarray): melody with n notes components:
                    1st note of melody,
                    array of relative pitches with shape (n-1,),
                    durations of notes with shape (n-1,).

    '''
    initial_dict = {k:f.findall(s) for k, f in features.items()}
    '''Example:
    {'ticks': ['220'],
     'qpm': ['120.0'],
     'total_time': ['3.5'],
     'pitch': ['63', '65', '72', '67', '72', '69'],
     'start_time': ['0.75', '1.0', '1.5', '1.75', '2.0'],
     'end_time': ['0.625', '1.0', '1.5', '1.75', '2.0', '3.5']}
    '''
    try:
        for k, v in initial_dict.items():
            if k in ['ticks', 'qpm', 'total_time']:
                initial_dict[k] = float(v[0])
            else:
                initial_dict[k] = np.array([float(x) for x in v])
        '''After:
        {'ticks': 220.0,
         'qpm': 120.0,
         'total_time': 3.5,
         'pitch': array([63., 65., 72., 67., 72., 69.]),
         'start_time': array([0.75, 1.  , 1.5 , 1.75, 2.  ]),
         'end_time': array([0.625, 1.   , 1.5  , 1.75 , 2.   , 3.5  ])
        '''
        # convert pitch from absolute to relative
        initial_dict['1st_note'] = initial_dict['pitch'][0] / 80.

        initial_dict['pitch'] = np.append(0, np.diff(initial_dict['pitch'])).astype(np.int8)
        if len(initial_dict['start_time']) < len(initial_dict['end_time']):
            initial_dict['start_time'] = np.append(0, initial_dict['start_time'])

    except IndexError:
        return -1, \
               np.append(0, np.repeat(np.cumsum(np.random.randint(32, 128, size=4)), 2))[:-1], \
               np.repeat(np.cumsum(np.random.randint(-128, 128, size=4)), 2)

    initial_dict['durations'] = initial_dict['end_time'] - initial_dict['start_time']
    initial_dict['durations'] /= 0.125
    initial_dict['durations'] = initial_dict['durations'].astype(np.uint8)

    return initial_dict['1st_note'], \
           np.append(0, np.repeat(np.cumsum(initial_dict['durations']), 2))[:-1], \
           np.repeat(np.cumsum(np.clip(initial_dict['pitch'], -15, 15)), 2)


def tanh(x):
    '''Apply tanh nonlinearity to input array.

            Args:
                x (numpy.ndarray): array.

            Returns:
                t (numpy.ndarray): array after applying tanh activation.

    '''
    exp = np.exp(x)
    exp_ = np.exp(-x)
    t = (exp - exp_)/(exp + exp_)
    return t


def justify(x_f, xp, xd, y_f, yp, yd, between='p', penalty_enabled=True):
    '''Compare two melodies of different size by its components and return distance value.

            Args:
                x_f (int): first note of the first melody.
                xp (numpy.ndarray): array of relative pitches with shape (n-1,).
                xd (numpy.ndarray): array of durations with shape (n-1,).
                y_f (int): first note of the second melody.
                yp (numpy.ndarray): array of relative pitches with shape (m-1,).
                yd (numpy.ndarray): array of durations with shape (m-1,).
                between (str, default is 'p'): if 'p' treats second melody as positive,
                    otherwise as negative.
                penalty_enabled (bool, default is True): if True then add to the resulting
                    distance additional term, when number of notes is not in range [3, 7]

            Returns:
                distance (float): distance between two melodies.

    '''
    if x_f == -1 or y_f == -1:
        return 100 if between == 'p' else -100
    lx, ly = xp.size, yp.size
    if (max(ly, lx) < 3 or max(ly, lx) > 7) and penalty_enabled:
        penalty = 5 if between == 'p' else -5
    else:
        penalty = 0
    if lx > ly:
        repeats = lx//ly+1
        yp = np.tile(yp, repeats)[:lx]
        yd = np.cumsum(np.tile(np.append(0, np.diff(yd)), repeats))[:lx]
    elif ly > lx:
        repeats = ly//lx+1
        xp = np.tile(xp, repeats)[:ly]
        xd = np.cumsum(np.tile(np.append(0, np.diff(xd)), repeats))[:ly]

    denom = np.repeat(np.log2(np.arange(2, xp.size//2+2)), 2)
    unnormalized_score = (np.abs(xp-yp)/(2.6)+np.log(1+np.abs(xd-yd)/(4)))/denom
    normalized_score = unnormalized_score / np.sum(1 / denom)
    return (np.mean(normalized_score) + np.abs(x_f-y_f)*4) + penalty


def visualize(W_b, counter, part):
    '''Checkpoints best results so far during search process on local drive.
    Computes AUC and generate melodies from embeddings.

            Args:
                W_b (numpy.ndarray): best found parameters so far.
                counter (int): iterations passed.
                part (str): directory to save.

            Returns:
                None.

    '''
    t = tanh(np.sum([(a * gaussians[j]).dot(x_test_origin.T).T for j, a in enumerate(W_b)],
                    axis=0)) * 2

    data = np.random.uniform(-0.01, 0.01, size=(t.shape[0], 512))
    data[:, significant_indexes] = t
    res = net.decode(data, length=16, temperature=0.001)
    res2 = [get_features_from_midi(str(g)) for g in res]

    pd1 = pd.DataFrame(res2, columns=['fst', 'd', 'p'])
    pd1['color'] = y_test_origin

    os.makedirs(os.path.join(f'mvae_{counter}_{part}'), exist_ok=True)
    joblib.dump(pd1, os.path.join(f'mvae_{counter}_{part}', f'decoded_colors_{counter}'))

    for i, note_seq in enumerate(res):
        os.makedirs(os.path.join(f'mvae_{counter}_{part}', str(y_test_origin[i])), exist_ok=True)
        music.sequence_proto_to_midi_file(note_seq, os.path.join(f'mvae_{counter}_{part}',
                                                                 str(y_test_origin[i]),
                                                                 f'{i}.midi'))
    probs, yprob = compute_probs(pd1, x_test_origin, y_test_origin)
    fpr, tpr, thresholds, auc = compute_metrics(probs, yprob)
    draw_roc(auc, fpr, tpr, thresholds, f'mvae_{counter}_{part}')
    draw_interdist(pd1, counter, f'mvae_{counter}_{part}')


def evaluate(params):
    '''Evaluate given parameters of transition matrix and returns triplet loss
    on custom distance functions.

            Args:
                params (numpy.ndarray): current parameters to evaluate.

            Returns:
                score (float): triplett loss calculated on custom melody distance function.

    '''
    global counter
    global best_score
    counter += 1
    print(counter)
    if ((counter+1) % 100) == 0:
        print('checkpoint...')
        try:
            opt = optimizer
            W_b = params[0]
            part = 'bayesian'
            joblib.dump(opt, f'opt_after_{counter}')
        except Exception as e:
            print(e)
            W_b = samples[np.argmin(scores)]
            part = 'random'
        visualize(W_b, counter, part)

    score = []
    for i in range(2):
        ts = time.time()
        triplets = get_batch_random(1024)
        triplets_tanh = np.concatenate([tanh(np.sum([(a*gaussians[j]).dot(x.T).T
                                                     for j, a in enumerate(params[0])],
                                                    axis=0)) * 2 for x in triplets])

        #data = np.random.uniform(-0.25, 0.25, size=(triplets[0].shape[0]*3, 512))
        #data[:, significant_indexes] = triplets_tanh
        data = triplets_tanh

        res = net.decode(data, length=16, temperature=0.001)
        res = [get_features_from_midi(str(g)) for g in res]  # r[0] = b1s, r[1] =  bds, r[2] = bps
        pd1 = pd.DataFrame(res[:triplets[0].shape[0]], columns=['a_fst', 'a_d', 'a_p'])

        pd1[['p_fst', 'p_d', 'p_p']] = pd.DataFrame(res[triplets[0].shape[0]:
                                                        triplets[0].shape[0]*2])
        pd1[['n_fst', 'n_d', 'n_p']] = pd.DataFrame(res[triplets[0].shape[0]*2:])
        a_columns = ['a_fst', 'a_p', 'a_d', 'p_fst', 'p_p', 'p_d']
        n_columns = ['a_fst', 'a_p', 'a_d', 'n_fst', 'n_p', 'n_d']
        pd1['ap'] = pd1.loc[:, a_columns].apply(lambda x: justify(x['a_fst'],
                                                                  x['a_p'],
                                                                  x['a_d'],
                                                                  x['p_fst'],
                                                                  x['p_p'],
                                                                  x['p_d']),
                                                axis=1)
        pd1['an'] = pd1.loc[:, n_columns].apply(lambda x: justify(x['a_fst'],
                                                                  x['a_p'],
                                                                  x['a_d'],
                                                                  x['n_fst'],
                                                                  x['n_p'],
                                                                  x['n_d'],
                                                                  between='n'),
                                                axis=1)
        score.append(np.sum(np.maximum(pd1.ap - pd1.an + 0.8, 0)))
        print('Time spent:', time.time() - ts)
        print('Median diff without margin:', np.mean(pd1.ap - pd1.an))
    if np.mean(score) < best_score:
        print("Found new best_score! Saving...")
        best_score = np.mean(score)
        joblib.dump(params[0], 'best_params100.gz', 3, 4)
    return np.mean(score)


if __name__ == '__main__':
    sys.path.append('run/')
    sys.path.append('./')
    Args = namedtuple('Args', ' '.join(['batch_size', 'beta', 'data_path', 'epochs', 'exp_path',
                                        'export_step', 'init_epoch', 'init_vis_epoch',
                                        'musicvae_config', 'musicvae_path', 'task']))
    args = Args(1024 * 3, 1.0, '', 200, '../exp/mnist_syn_small', 5, 101, 0,
                'cat-mel_2bar_big', '../models/cat-mel_2bar_big.ckpt', 'mnist')
    ##########     Initialization   ########
    with tf.device('/gpu:0'):
        config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
        with tf.compat.v1.Session(config=config) as sess:
            net = TrainedModel(
                config=ccopy.deepcopy(ccopy.deepcopy(configs.CONFIG_MAP[args.musicvae_config])),
                batch_size=args.batch_size,
                checkpoint_dir_or_path=args.musicvae_path)
    counter = 0
    nb_classes = 9
    data = joblib.load('embed.gz')
    dataset_train = []
    dataset_test = []
    x_test_origin = []
    y_test_origin = []
    for i in tqdm(range(9)):
        dataset_train.append(data[i][:int(data[i].shape[0] * 0.95)])
        dataset_test.append(data[i][int(data[i].shape[0] * 0.95):])
        for x in dataset_test[i]:
            x_test_origin.append(x)
            y_test_origin.append(i)
    x_test_origin = np.array(x_test_origin)
    y_test_origin = np.array(y_test_origin)
    if os.path.exists('gaussians100.dat'):
        gaussians = joblib.load('gaussians100.dat')
    else:
        gaussians = np.random.randn(100, 512, 9)
        joblib.dump(gaussians, 'gaussians100.dat')
    # significant_indexes = [12, 182, 471, 455, 371, 62, 493, 208,
    #            59, 195, 271, 437, 262, 121, 216, 245,
    #            226, 377, 167, 14, 280, 282, 297, 363,
    #            24, 81]
    significant_indexes = np.arange(512)
    ####$$$$$$$$### RANDOM SEARCH #############
    samples = []
    scores = []
    best_score = np.inf
    for i in tqdm(range(len(significant_indexes)*4)):
        b = np.random.uniform(-0.07, 0.07, size=(len(gaussians)))
        samples.append(b)
        scores.append(evaluate(b.reshape(1, -1)))
    joblib.dump(samples, 'samples_weights100.gz', 3, 4)
    joblib.dump(scores, 'samples_rates100.gz', 3, 4)
    #################### END ####################
    #####$$$$$$### BAYESIAN SEARCH #############
    counter = 0
    samples = np.array(joblib.load('samples_weights100.gz'))
    scores = np.array(joblib.load('samples_rates100.gz'))

    print(scores.shape, scores[:, None].shape)
    bounds = [{'name': 'w_{0:03d}'.format(i),
               'type': 'continuous',
               'domain': (-0.07, 0.07)}
              for i in range(len(gaussians))]
    optimizer = GPyOpt.methods.BayesianOptimization(f=evaluate, domain=bounds,
                                                    model_type='sparseGP',
                                                    acquisition_type='LCB',
                                                    acquisition_par=0.3,
                                                    exact_eval=False,  # Since we are not sure
                                                    X=samples[scores.argsort()[:100]],
                                                    Y=scores[scores.argsort()[:100]][:, None],
                                                    de_duplication=True,
                                                    maximize=False)
    print(scores.min())
    print(scores[scores.argsort()[:2]])
    optimizer.run_optimization(max_iter=len(bounds)*12, eps=-1)
    print(optimizer.x_opt)
    print(evaluate(optimizer.x_opt[None, :]))
    joblib.dump(optimizer.x_opt, 'best_weights__.gz', 3, 4)
    print(optimizer.fx_opt)
    joblib.dump(optimizer, 'opt_final')
    for i in range(0, 15):
        visualize(samples[scores.argsort()[i]], f'{i}99', 'top')
    optimizer = joblib.load('opt_final')
    visualize(optimizer.x_opt, 0, 'final')
