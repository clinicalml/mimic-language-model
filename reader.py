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


"""Utilities for parsing text files.
Based on the TensorFlow tutorial for building a PTB LSTM model.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import os.path
from os.path import join as pjoin
import re

import numpy as np
import tensorflow as tf
import nltk

from mimictools import utils as mutils


_fix_re = re.compile(r"[^a-z0-9/'?.,-]+")
_num_re = re.compile(r'[0-9]+')
_dash_re = re.compile(r'-+')


def _fix_word(word):
    word = word.lower()
    word = _fix_re.sub('-', word).strip('-')
    word = _num_re.sub('#', word)
    word = _dash_re.sub('-', word)
    return word


def ptb_iterator(raw_data, num_steps, config):
    with open(pjoin(config.mimicpk_path, 'vocab_fd.pk'), 'rb') as f:
        vocab_fd = pickle.load(f)
    vocab = vocab.keys()
    vocab.insert(0, config.EOS) # end of sentence
    vocab.insert(1, config.UNK) # unknown
    vocab_lookup = {word: idx for (idx, word) in enumerate(vocab)}

    epoch = 1
    split = 0
    while True:
        notes_file = pjoin(config.mimicsp_path, '%02d/NOTEEVENTS_DATA_TABLE.csv' % (split,))
        if os.path.isfile(notes_file):
            print 'Epoch', epoch, ' File', notes_file
            raw_data = []
            for _, raw_text in mutils.mimic_data([notes_file], replace_anon='_'):
                sentences = nltk.sent_tokenize(raw_text)
                for sent in sentences:
                    words = [_fix_word(w) for w in nltk.word_tokenize(sent)
                                            if any(c.isalpha() or c.isdigit()
                                                for c in w)]
                    finalwords = []
                    for word in words:
                        if not word: continue
                        if word in vocab:
                            finalwords.append(vocab_lookup[word])
                        else:
                            finalwords.append(1) # vocab_lookup[config.UNK]
                    finalwords.append(0) # vocab_lookup[config.EOS]
                    raw_data.extend(finalwords)

            raw_data = np.array(raw_data, dtype=np.int32)
            data_len = len(raw_data)
            batch_len = data_len // config.batch_size
            data = np.zeros([config.batch_size, batch_len], dtype=np.int32)
            for i in range(batch_size):
                data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

            epoch_size = (batch_len - 1) // num_steps
            if epoch_size == 0:
                raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

            for i in range(epoch_size): # XXX check about the last tokens after the data stream has ended
                x = data[:, i*num_steps:(i+1)*num_steps]
                y = data[:, i*num_steps+1:(i+1)*num_steps+1]
                yield (x, y)

        split += 1
        if split >= 100:
            split = 0
            epoch += 1
