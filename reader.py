# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#==============================================================================


"""Utilities for parsing MIMIC data.
Based on the TensorFlow tutorial for building a PTB LSTM model.
"""
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
            self.aux_set = {feat: set(vals) for (feat, vals) in self.aux_list.items()}
            self.aux_lookup = {feat: {val: idx for (idx, val) in enumerate(vals)}
                                     for (feat, vals) in self.aux_list.items()}
            for (k,v) in self.aux_list.items():
                print k, len(v)
            print 'Auxiliary vocab loaded.'


def mimic_iterator(config):
    splits = range(100)
    random.shuffle(splits)
    for split in splits:
        notes_file = pjoin(config.data_path, 'notes_%02d.pk' % (split,))
        if os.path.isfile(notes_file):
            print 'Loading data split', split
            with open(notes_file, 'rb') as f:
                data = pickle.load(f)
            raw_data = []
            for note in data:
                values = {}
                (text, values['gender'], values['has_dod'], values['has_icu_stay'], \
                 values['admission_type'], values['diagnoses'], values['procedures'], \
                 values['labs'], values['prescriptions']) = note
                if config.conditional:
                    raw_data.extend([[w] + [values[feat] for feat in config.fixed_len_features[1:]] for w in text])
                else:
                    raw_data.extend([[w] for w in text])
            # TODO the variable length stuff
            print 'Loaded data split', split, ', processing.'
            data_len = len(raw_data)
            raw_data = np.array(raw_data, dtype=np.int32)
            batch_len = data_len // config.batch_size
            data = np.zeros([config.batch_size, batch_len, config.data_size], dtype=np.int32)
            for i in xrange(config.batch_size):
                data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

            epoch_size = (batch_len - 1) // config.num_steps
            if epoch_size == 0:
                raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

            print 'Data split', split, 'ready.'
            for i in xrange(epoch_size):
                x = data[:, i*config.num_steps:(i+1)*config.num_steps]
                y = data[:, i*config.num_steps+1:(i+1)*config.num_steps+1, 0]
                yield (x, y)
