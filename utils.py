import itertools
import random

import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np


class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'


def print_color(s, color=None):
    if color:
        print color + str(s) + Colors.ENDC,
    else:
        print s,


def grouper(n, iterable, fillvalue=None):
    args = [iter(iterable)] * n
    return itertools.izip_longest(*args, fillvalue=fillvalue)


def subset(seq, k):
    if not 0 <= k <= len(seq):
        for e in seq:
            yield e
    else:
        numbersPicked = 0
        for i, number in enumerate(seq):
            prob = (k-numbersPicked) / (len(seq)-i)
            if random.random() < prob:
                yield number
                numbersPicked += 1


def l1_norm(tensor):
    return tf.reduce_sum(tf.abs(tensor))


def inspect_feature_sparsity(feat, embedding, config, vocab, verbose=False, graph=True):
    print '\n\n' + feat + '\n'
    vocab_size, dims = embedding.shape
    if verbose and vocab_size < 10 and dims < 10:
        for i in xrange(vocab_size):
            try:
                toprint = vocab.aux_list[feat][i]
            except:
                toprint = i
            print str(toprint).ljust(20), embedding[i]
    if graph and feat in config.var_len_features:
        embedding = sorted(np.abs(embedding.flatten()).tolist(), reverse=True)
        print len(embedding)
        plt.figure(figsize=(14,10))
        plt.plot(embedding)
        plt.axis([0, len(embedding)-1, 0.0, embedding[0]])
        plt.title('Sparsity pattern for '+feat)
        plt.ylabel('Absolute values')
        plt.show()


def inspect_sparsity(session, m, config, vocab, saver):
    with tf.device("/cpu:0") and tf.variable_scope("model", reuse=True):
        for i, (feat, dims) in enumerate(config.mimic_embeddings.items()):
            if dims <= 0: continue
            try:
                vocab_aux = len(vocab.aux_list[feat])
            except KeyError:
                vocab_aux = 2 # binary
            with tf.device("/cpu:0"):
                vocab_dims = vocab_aux
                if feat in config.var_len_features:
                    vocab_dims -= 1
                embedding = tf.get_variable("struct_embedding."+feat, [vocab_dims,
                                                                    config.mimic_embeddings[feat]],
                                           initializer=tf.truncated_normal_initializer(stddev=0.1))
                inspect_feature_sparsity(feat, embedding.eval(), config, vocab)
