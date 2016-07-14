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

"""Based on the TensorFlow tutorial for building a PTB LSTM model."""
from __future__ import division

import time

import numpy as np
import tensorflow as tf

from config import Config
import reader

flags = tf.flags
logging = tf.logging

FLAGS = flags.FLAGS


class LMModel(object):
    """The language model."""

    def __init__(self, is_training, config, vocab):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        size = config.hidden_size
        vocab_size = config.vocab_size

        self.input_data = tf.placeholder(tf.int32, [batch_size, num_steps, config.data_size])
        self.targets = tf.placeholder(tf.int32, [batch_size, num_steps])

        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size)
        if is_training and config.keep_prob < 1:
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
                    lstm_cell, output_keep_prob=config.keep_prob)
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers)

        self.initial_state = cell.zero_state(batch_size, tf.float32)

        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [vocab_size, config.learn_emb_size])
            if config.pretrained_emb:
                cembedding = tf.constant(vocab.embeddings, dtype=embedding.dtype,
                                         name="pre_embedding")
                embedding = tf.concat(1, [embedding, cembedding])
            word_embedding = tf.nn.embedding_lookup(embedding,
                                                    tf.squeeze(tf.slice(self.input_data,
                                                                        [0,0,0], [-1,-1,1]), [2]))
            inputs = word_embedding

        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)

        inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(1, num_steps, inputs)]
        outputs, state = tf.nn.rnn(cell, inputs, initial_state=self.initial_state)

        output = tf.reshape(tf.concat(1, outputs), [-1, size])
        softmax_w = tf.get_variable("softmax_w", [size, vocab_size])
        softmax_b = tf.get_variable("softmax_b", [vocab_size])
        logits = tf.matmul(output, softmax_w) + softmax_b
        loss = tf.nn.seq2seq.sequence_loss_by_example(
                [logits],
                [tf.reshape(self.targets, [-1])],
                [tf.ones([batch_size * num_steps])])
        self.cost = cost = tf.reduce_sum(loss) / batch_size
        self.final_state = state

        if not is_training:
            return

        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                          config.max_grad_norm)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))


def run_epoch(session, m, eval_op, config, verbose=False):
    """Runs the model on the given data."""
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = m.initial_state.eval()
    for step, (x, y) in enumerate(reader.mimic_iterator(config)):
        cost, state, _ = session.run([m.cost, m.final_state, eval_op],
                                     {m.input_data: x,
                                      m.targets: y,
                                      m.initial_state: state})
        costs += cost
        iters += m.num_steps

        if verbose and step % config.print_every == 0:
            print("%d  perplexity: %.3f speed: %.0f wps" %
                        (step, np.exp(costs / iters),
                         iters * m.batch_size / (time.time() - start_time)))

    return np.exp(costs / iters)


def main(_):
    config = Config()
    if config.conditional:
        print 'Training a conditional language model for MIMIC'
    else:
        print 'Training an unconditional language model for MIMIC'
    vocab = reader.Vocab(config)

    config_proto = tf.ConfigProto()
    config_proto.gpu_options.allow_growth = True
    with tf.Graph().as_default(), tf.Session(config=config_proto) as session:
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            m = LMModel(is_training=True, config=config, vocab=vocab)
        tf.initialize_all_variables().run()

        for i in range(config.max_epoch):
            #lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
            m.assign_lr(session, config.learning_rate) #* lr_decay)

            print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
            train_perplexity = run_epoch(session, m, m.train_op, config,
                                         verbose=True)
            print("Epoch: %d Train Perplexity: %.3f" % (i + 1,
                                                        train_perplexity))


if __name__ == "__main__":
    tf.app.run()
