import itertools
from os.path import join as pjoin
import random
import cPickle as pickle
import collections

import matplotlib
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


def l2_norm(tensor):
    return tf.sqrt(tf.reduce_sum(tf.mul(tensor, tensor)))


def _inspect_losses(x, y, config, vocab, loss, aux, aux_len, dicts):
    print_color('[', Colors.HEADER)
    for i in range(config.num_steps // 2):
        print vocab.vocab_list[x[i]],
    print_color(vocab.vocab_list[y], Colors.OKGREEN)
    for i in range(config.num_steps // 2, config.num_steps):
        print vocab.vocab_list[x[i]],
    print_color(']', Colors.HEADER)
    print
    for k, v, g in loss:
        print ("perp %.5f, prob %.5f:" % (np.exp(k), np.exp(-k))),
        if v == 'all': color = Colors.OKGREEN
        elif v == 'none' or v == 'unconditional': color = Colors.FAIL
        elif v.startswith('only_'): color = Colors.OKBLUE
        else: color = Colors.WARNING
        print_color(v.ljust(20), color)
        print 'gate min %.6f, max %.6f, avg %.6f, std %.6f' % (np.min(g), np.max(g),
                                                               np.mean(g), np.std(g))
    print
    for feat, v in aux.items():
        print feat
        if feat in config.var_len_features:
            print ', '.join([str(dicts.get(feat, {}).get(vocab.aux_list[feat][val], val))
                                for val in v[:aux_len[feat]]])
        else:
            try:
                print vocab.aux_list.get(feat, [])[v[0]]
            except IndexError:
                if feat == 'gender':
                    if v[0] == 1: print 'FEMALE'
                    else: print 'MALE'
                else:
                    print v[0]
        print
    print


losses_buffer = []

def inspect_losses(xs, ys, config, vocab, losses, aux, aux_len, dicts, max_minperp=150.0,
                   buffer_size=1000):
    import nltk
    global losses_buffer
    for i, (x, y, loss) in enumerate(zip(xs, ys, losses)):
        word = vocab.vocab_list[y]
        if word == '|' or word == '+' or '#' in word or \
                word in nltk.corpus.stopwords.words('english'):
            continue
        #valid = True
        #for context in x:
        #    word = vocab.vocab_list[context]
        #    if word == '|' or word == '+' or '#' in word:
        #        valid = False
        #        break
        #if not valid: continue
        loss = sorted(loss, key=lambda x:x[0])
        la = np.exp(np.array([l[0] for l in loss]))
        if np.amin(la) > max_minperp: continue
        stdev = np.std(la / np.amax(la))
        d = {k:v for v,k,_ in loss}
        for k in ['unconditional', 'none']:
            try:
                if d['all'] > d[k]:
                    stdev = -stdev
                    break
            except KeyError:
                pass
        aux_ = {k:v[i] for k,v in aux.items()}
        aux_len_ = {k:v[i] for k,v in aux_len.items()}
        losses_buffer.append((stdev, x, y, loss, aux_, aux_len_))
        if buffer_size > 0 and len(losses_buffer) >= buffer_size:
            losses_buffer = sorted(losses_buffer, key=lambda x:x[0])
            for s, x_, y_, loss_, aux_, aux_len_ in losses_buffer:
                _inspect_losses(x_, y_, config, vocab, loss_, aux_, aux_len_, dicts)
            losses_buffer = []
            print 'Press enter to continue ...'
            raw_input()


def make_struct_mappings(dicts):
    ret = {}
    for k, v in dicts.items():
        if k == 'D_LABITEMS_DATA_TABLE.csv':
            superkey = 'labs'
            key = 'ITEMID'
            value = ['CATEGORY', 'LABEL']
        elif k == 'D_ICD_DIAGNOSES_DATA_TABLE.csv':
            superkey = 'diagnoses'
            key = 'ICD9_CODE'
            value = ['SHORT_TITLE']
        elif k == 'D_ICD_PROCEDURES_DATA_TABLE.csv':
            superkey = 'procedures'
            key = 'ICD9_CODE'
            value = ['SHORT_TITLE']
        mapping = {}
        for _, val in v.items():
            try:
                mapping[val[key]] = ' | '.join([val[s] for s in value])
            except KeyError:
                pass
        ret[superkey] = mapping
    return ret


def inspect_compare(config, vocab):
    with open(config.load_uncond_results, 'rb') as f:
        uncond = pickle.load(f)
    with open(config.load_cond_results, 'rb') as f:
        cond = pickle.load(f)
    with open(pjoin(config.data_path, 'dicts.pk'), 'rb') as f:
        dicts = pickle.load(f)
    dicts = make_struct_mappings(dicts)
    for (x, y, cond_losses, aux, aux_len), (x_, y_, uncond_losses) in zip(cond, uncond):
        assert np.all(x == x_)
        assert np.all(y == y_)
        for i in xrange(config.batch_size):
            cond_losses[i].extend(uncond_losses[i])
        inspect_losses(x, y, config, vocab, cond_losses, aux, aux_len, dicts)


def inspect_feature_embs(feat, embedding, config, vocab, dicts, fd, topk=2500):
    shift = 0
    if feat == 'words':
        vocablist = vocab.vocab_list
    elif dicts:
        if feat in config.var_len_features:
            shift = 1
        try:
            vocablist = vocab.aux_list[feat]
        except KeyError:
            return
    else:
        return

    from tsne import bh_sne
    print '\n' + feat
    perp = 10
    W, H = 90, 90
    if len(vocablist) < 5:
        perp = 1
        W, H = 5, 5

    if feat == 'words':
        keep = [k for k,v in sorted(fd.items(), key=lambda x: x[1], reverse=True)[:topk]]
        keep = list(set([vocab.vocab_lookup.get(k, 0) for k in keep]))
    else:
        keep = list(set([k for k,v in sorted(fd.items(), key=lambda x: x[1],
                                                 reverse=True)[:topk]]))
        keep = [k for k in keep if k >= shift]
    keep = np.array(keep)

    embedding = embedding[keep-shift]
    embedding = bh_sne(embedding.astype(np.float64), perplexity=perp)
    x = embedding[:, 0]
    y = embedding[:, 1]

    print 'Preparing figure'
    cmap = matplotlib.cm.get_cmap('Greys')
    plt.figure(figsize=(W, H))
    if fd:
        norm = matplotlib.colors.LogNorm(vmin=1, vmax=max(fd.values()))
    z = []
    for i in range(len(keep)):
        if fd:
            if feat == 'words':
                freq = fd.get(vocablist[keep[i]], None)
            else:
                freq = fd.get(keep[i], None)
            if freq and freq > 0:
                freq = norm(freq)
            else:
                freq = 0.0
        else:
            freq = 1.0
        txt = vocablist[keep[i]]
        txt = dicts.get(txt, txt)
        z.append(freq)
        color = cmap(max(freq, 0.25))
        plt.text(x[i], y[i], txt, fontsize=7, color=color)
    plt.scatter(x, y, c=np.array(z), vmin=0.0, vmax=1.0, cmap=cmap)

    print 'Saving figure'
    plt.savefig(pjoin('figures', feat+'.png'), dpi=110)


def inspect_embs(session, m, config, vocab):
    with tf.device("/cpu:0") and tf.variable_scope("model", reuse=True):
        if not config.struct_only:
            with open(pjoin(config.data_path, 'vocab_fd.pk'), 'rb') as f:
                fd = pickle.load(f)
            word_embeddings = tf.get_variable("word_embedding", [config.vocab_size,
                                                                 config.word_emb_size])
            inspect_feature_embs('words', word_embeddings.eval(), config, vocab, {}, fd)
        if config.conditional:
            with open(pjoin(config.data_path, 'dicts.pk'), 'rb') as f:
                dicts = pickle.load(f)
            dicts = make_struct_mappings(dicts)
            with open(pjoin(config.data_path, 'aux_cfd.pk'), 'rb') as f:
                cfd = pickle.load(f)
            for i, (feat, dims) in enumerate(config.mimic_embeddings.items()):
                if dims <= 0: continue
                try:
                    vocab_aux = len(vocab.aux_list[feat])
                except KeyError:
                    vocab_aux = 2 # binary
                vocab_dims = vocab_aux
                if feat in config.var_len_features:
                    vocab_dims -= 1
                embedding = tf.get_variable("struct_embedding."+feat,
                                            [vocab_dims, config.mimic_embeddings[feat]])
                inspect_feature_embs(feat, embedding.eval(), config, vocab, dicts.get(feat, {}),
                                     cfd.get(feat, {}))


#XXX Unused
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


#XXX Unused
def inspect_sparsity(session, m, config, vocab):
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
