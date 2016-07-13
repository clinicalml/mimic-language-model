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
        print 'Vocab loaded, size:', config.vocab_size


def mimic_iterator(config):
    for split in xrange(100):
        notes_file = pjoin(config.data_path, 'notes_%02d.pk' % (split,))
        if os.path.isfile(notes_file):
            print 'Loading data split', split
            with open(notes_file, 'rb') as f:
                raw_data = pickle.load(f)
            print 'Loaded data split', split, ', processing.'
            data_len = len(raw_data)
            raw_data = np.array(raw_data, dtype=np.int32)
            batch_len = data_len // config.batch_size
            data = np.zeros([config.batch_size, batch_len], dtype=np.int32)
            for i in xrange(config.batch_size):
                data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

            epoch_size = (batch_len - 1) // config.num_steps
            if epoch_size == 0:
                raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

            print 'Data split', split, ' ready.'
            for i in xrange(epoch_size):
                x = data[:, i*config.num_steps:(i+1)*config.num_steps]
                y = data[:, i*config.num_steps+1:(i+1)*config.num_steps+1]
                yield (x, y)
