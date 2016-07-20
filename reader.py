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


class Vocab(object):
    def __init__(self, config):
        print 'Loading vocab ...'
        with open(pjoin(config.data_path, 'vocab.pk'), 'rb') as f:
            self.vocab_list = pickle.load(f)
        self.vocab_lookup = {word: idx for (idx, word) in enumerate(self.vocab_list)}
        self.vocab_set = set(self.vocab_list) # for faster checks
        config.vocab_size = len(self.vocab_list)
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


def mimic_iterator(config, vocab):
    splits = range(100)
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

            notes_count = len(raw_data)
            batch_len = ((notes_count - 1) // config.batch_size) + 1
            for batch in xrange(batch_len):
                batch_data = raw_data[config.batch_size * batch : config.batch_size * (batch + 1)]
                if config.conditional:
                    batch_aux_data = {}
                    for (feat, vals) in raw_aux_data.items():
                        batch_aux_data[feat] = vals[config.batch_size * batch : \
                                                    config.batch_size * (batch + 1)]
                max_note_len = max(len(note) for note in batch_data)
                epoch_size = ((max_note_len - 2) // config.num_steps) + 1
                data = np.zeros([config.batch_size, epoch_size * config.num_steps + 1],
                                dtype=np.int32)
                mask = np.zeros([config.batch_size, epoch_size * config.num_steps + 1],
                                dtype=np.float32)
                for i, iter_data in enumerate(batch_data):
                    data[i, 0:len(iter_data)] = iter_data
                    mask[i, 0:len(iter_data)] = 1.0
                aux_data = {}
                aux_data_len = {}
                if config.conditional:
                    for feat, vals in batch_aux_data.items():
                        if feat in config.fixed_len_features:
                            max_struct_len = 1
                        else:
                            max_struct_len = max(len(v) for v in vals)
                        aux_data[feat] = np.zeros([config.batch_size, max_struct_len],
                                                  dtype=np.int32)
                        aux_data_len[feat] = np.zeros([config.batch_size], dtype=np.int32)
                        for i, iter_data in enumerate(vals):
                            if feat in config.fixed_len_features:
                                aux_data[feat][i, 0] = iter_data
                                aux_data_len[feat][i] = 1
                            else:
                                aux_data[feat][i, 0:len(iter_data)] = iter_data
                                aux_data_len[feat][i] = len(iter_data)

                new_batch = True
                for i in xrange(epoch_size):
                    x = data[:, i*config.num_steps:(i+1)*config.num_steps]
                    y = data[:, i*config.num_steps+1:(i+1)*config.num_steps+1]
                    m = mask[:, i*config.num_steps+1:(i+1)*config.num_steps+1]
                    yield (x, y, m, aux_data, aux_data_len, new_batch)
                    new_batch = False
