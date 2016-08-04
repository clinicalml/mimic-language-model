from __future__ import division

import collections
import os
import os.path
from os.path import join as pjoin
import random
import re
import cPickle as pickle

import numpy as np
import tensorflow as tf
import nltk

import utils


class Vocab(object):
    def __init__(self, config):
        random.seed(0) # make deterministic
        print 'Loading vocab ...'
        with open(pjoin(config.data_path, 'vocab.pk'), 'rb') as f:
            self.vocab_list = pickle.load(f)
        self.vocab_lookup = {word: idx for (idx, word) in enumerate(self.vocab_list)}
        self.vocab_set = set(self.vocab_list) # for faster checks
        config.vocab_size = len(self.vocab_list)
        self.shuffled_indices = range(config.vocab_size)
        random.shuffle(self.shuffled_indices) # for HSM
        if config.pretrained_emb:
            print 'Loading pretrained embeddings ...'
            self.embeddings = None
            with open(pjoin(config.data_path, 'vocab_embeddings'), 'r') as f:
                for line in f:
                    tokens = line.split()
                    word = tokens[0]
                    if word in self.vocab_set:
                        emb = [float(t) for t in tokens[1:]]
                        if self.embeddings is None:
                            self.embeddings = np.zeros([config.vocab_size, len(emb)])
                        self.embeddings[self.vocab_lookup[word]] = emb
        print 'Vocab loaded, size:', config.vocab_size
        if config.conditional:
            print 'Loading vocab for auxiliary data ...'
            with open(pjoin(config.data_path, 'vocab_aux.pk'), 'rb') as f:
                # loads labs, diagnoses, procedures, prescriptions
                self.aux_list = pickle.load(f)
            self.aux_list['admission_type'] = ['ELECTIVE', 'URGENT', 'NEWBORN', 'EMERGENCY']
            for feat in config.var_len_features:
                self.aux_list[feat].insert(0, None) # padding value
            self.aux_set = {feat: set(vals) for (feat, vals) in self.aux_list.items()}
            self.aux_lookup = {feat: {val: idx for (idx, val) in enumerate(vals)}
                                     for (feat, vals) in self.aux_list.items()}
            for (k,v) in self.aux_list.items():
                print k, len(v)
            print 'Auxiliary vocab loaded.'


def _mimic_iterator_unbuffered(config, vocab):
    if config.training:
        splits = config.training_splits
    else:
        splits = config.testing_splits
    random.shuffle(splits)

    for split in splits:
        notes_file = pjoin(config.data_path, 'notes_%02d.pk' % (split,))
        if os.path.isfile(notes_file):
            print 'Loading data split', split
            with open(notes_file, 'rb') as f:
                data = pickle.load(f)
            raw_data = []
            if config.conditional:
                raw_aux_data = collections.defaultdict(list)
            for note in data:
                values = {}
                (text, values['gender'], values['has_dod'], values['has_icu_stay'], \
                 values['admission_type'], values['diagnoses'], values['procedures'], \
                 values['labs'], values['prescriptions']) = note
                if len(text) > 1:
                    raw_data.append(text)
                    if config.conditional:
                        for (feat, dims) in config.mimic_embeddings.items():
                            if dims > 0:
                                raw_aux_data[feat].append(values[feat])
            print 'Loaded data split', split

            batch_len = ((len(raw_data) - 1) // config.batch_size) + 1
            pad_count = (batch_len * config.batch_size) - len(raw_data)
            indices = range(batch_len)
            random.shuffle(indices)

            raw_data = [[] for _ in range(pad_count)] + raw_data
            grouped_raw_data = [group for group in utils.grouper(config.batch_size, raw_data)]
            grouped_raw_data = [grouped_raw_data[i] for i in indices]
            raw_data = [note for group in grouped_raw_data for note in group]
            if config.conditional:
                for k, v in raw_aux_data.items():
                    if k in config.fixed_len_features:
                        v = [0 for _ in range(pad_count)] + v
                    else:
                        v = [[] for _ in range(pad_count)] + v
                    ls = [group for group in utils.grouper(config.batch_size, v)]
                    ls = [ls[i] for i in indices]
                    raw_aux_data[k] = [data for group in ls for data in group]

            for batch in xrange(batch_len):
                batch_data = raw_data[config.batch_size * batch : config.batch_size * (batch + 1)]
                if config.conditional:
                    batch_aux_data = {}
                    for (feat, vals) in raw_aux_data.items():
                        batch_aux_data[feat] = vals[config.batch_size * batch : \
                                                    config.batch_size * (batch + 1)]
                if config.recurrent:
                    max_note_len = max(len(note) for note in batch_data)
                    epoch_size = ((max_note_len - 2) // config.num_steps) + 1
                    data = np.zeros([config.batch_size, epoch_size * config.num_steps + 1],
                                    dtype=np.int32)
                else:
                    min_note_len = min(len(note) for note in batch_data)
                    data = np.zeros([config.batch_size, min_note_len], dtype=np.int32)
                    epoch_size = min_note_len - config.num_steps # this can become negative!
                    if epoch_size <= 0:
                        continue
                if config.recurrent:
                    mask = np.zeros([config.batch_size, epoch_size * config.num_steps + 1],
                                    dtype=np.float32)
                for i, iter_data in enumerate(batch_data):
                    if config.recurrent:
                        data[i, 0:len(iter_data)] = iter_data
                        mask[i, 0:len(iter_data)] = 1.0
                    else:
                        data[i, 0:min_note_len] = iter_data[:min_note_len]
                aux_data = {}
                if config.conditional:
                    for feat, vals in batch_aux_data.items():
                        if feat in config.fixed_len_features:
                            max_struct_len = 1
                        else:
                            max_struct_len = max(len(v) for v in vals)
                        aux_data[feat] = np.zeros([config.batch_size, max_struct_len],
                                                  dtype=np.int32)
                        for i, iter_data in enumerate(vals):
                            if feat in config.fixed_len_features:
                                aux_data[feat][i, 0] = iter_data
                            else:
                                aux_data[feat][i, 0:len(iter_data)] = iter_data

                new_batch = True
                epochs = range(epoch_size)
                if not config.recurrent:
                    epochs = [e for e in utils.subset(epochs, config.samples_per_note)]
                for i in epochs:
                    if config.recurrent:
                        x = data[:, i*config.num_steps:(i+1)*config.num_steps]
                        y = data[:, i*config.num_steps+1:(i+1)*config.num_steps+1]
                        m = mask[:, i*config.num_steps+1:(i+1)*config.num_steps+1]
                    else:
                        x = np.concatenate([data[:, i:(i+int(config.num_steps/2))],
                                   data[:, (i+1+int(config.num_steps/2)):i+1+config.num_steps]], 1)
                        y = data[:, i+int(config.num_steps/2)]
                        m = None
                    yield (x, y, m, aux_data, new_batch)
                    new_batch = False


def mimic_iterator(config, vocab):
    random.seed(0) # make deterministic
    if config.recurrent:
        yield _mimic_iterator_unbuffered(config, vocab)
    else:
        batches = []
        for data in _mimic_iterator_unbuffered(config, vocab):
            batches.append(data)
            size = len(batches)
            if size >= config.data_rand_buffer:
                j = random.randint(0, size-1)
                yield batches.pop(j)
        random.shuffle(batches)
        for batch in batches:
            yield batch
