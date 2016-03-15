from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple

import numpy as np
import os
import tensorflow as tf

from tensorflow.models.rnn import rnn

Config = namedtuple('Config',
                    ['batch_size', 'num_steps', 'hidden_size', 'embedding_dim',
                     'dropout_keep_prob', 'charset_size', 'num_layers', 'max_grad_norm'])

class CharRNN(object):

  def __init__(self, stage, config):
    self._batch_size = config.batch_size if not stage == 'infer' else 1
    self._num_steps = config.num_steps if not stage == 'infer' else 1
    self._hidden_size = config.hidden_size
    self._embedding_dim = config.embedding_dim
    self._dropout_keep_prob = config.dropout_keep_prob
    self._num_layers = config.num_layers
    self._charset_size = config.charset_size
    self._stage = stage

    self._input_data = tf.placeholder(tf.int32, [self._batch_size,
                                                 self._num_steps])
    self._targets = tf.placeholder(tf.int32, [self._batch_size,
                                              self._num_steps])
    lstm_cell_input = tf.nn.rnn_cell.LSTMCell(self._hidden_size,
                                              self._embedding_dim)
    lstm_cell = tf.nn.rnn_cell.LSTMCell(self._hidden_size,
                                        self._hidden_size)
    if stage == 'train' and self._dropout_keep_prob < 1:
      lstm_cell_input = tf.nn.rnn_cell.DropoutWrapper(
        lstm_cell_input, output_keep_prob=self._dropout_keep_prob)
      lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
        lstm_cell, output_keep_prob=self._dropout_keep_prob)
    self._cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_input] +
                                             [lstm_cell] * self._num_layers)

    self._initial_state = self._cell.zero_state(self._batch_size, tf.float32)

    with tf.device('/cpu:0'):
      self._embedding = tf.get_variable('embedding', [self._charset_size,
                                                      self._embedding_dim])
      inputs = tf.nn.embedding_lookup(self._embedding, self._input_data)

    inputs = [tf.squeeze(input_, [1])
          for input_ in tf.split(1, self._num_steps, inputs)]
    outputs, state = rnn.rnn(self._cell, inputs,
                 initial_state=self._initial_state)

    output = tf.reshape(tf.concat(1, outputs), [-1, self._hidden_size])
    self._output_shape = tf.shape(output)
    self._softmax_w = tf.get_variable('softmax_w',
                                      [self._hidden_size, self._charset_size])
    self._softmax_b = tf.get_variable('softmax_b', [self._charset_size])
    self._logits = tf.matmul(output, self._softmax_w) + self._softmax_b
    self._probability = tf.nn.softmax(self._logits)
    loss = tf.nn.seq2seq.sequence_loss_by_example(
      [self._logits],
      [tf.reshape(self._targets, [-1])],
      [tf.ones([self._batch_size * self._num_steps])], self._charset_size)
    self._cost = cost = tf.reduce_sum(loss) / self._batch_size
    self._final_state = state

    pred = tf.argmax(self._logits, 1)
    labels = tf.cast(tf.reshape(self._targets, [-1]), tf.int64)
    self._misclass = 1 - tf.reduce_mean(tf.cast(tf.equal(pred, labels),
                                                tf.float32))

    if stage != 'train':
      return

    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                      config.max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(self.lr)
    self._train_op = optimizer.apply_gradients(zip(grads, tvars))

  def sample(self, sess, char_to_id, num_steps, seed):
    state = self._cell.zero_state(1, tf.float32).eval()
    id_to_char = {v:k for k, v in char_to_id.iteritems()}
    res = seed

    def weighted_pick(weights):
      t = np.cumsum(weights)
      s = np.sum(weights)
      return(int(np.searchsorted(t, np.random.rand(1)*s)))

    for char in seed[:-1]:
      x = np.zeros([1,1])
      x[0, 0] = char_to_id[char]
      feed = {self.input_data: x, self.initial_state: state}
      [state] = sess.run([self.final_state], feed)

    char = seed[-1]
    for _ in xrange(num_steps):
      x = np.zeros((1, 1))
      x[0, 0] = char_to_id[char]
      feed = {self.input_data: x, self.initial_state:state}
      [prob, state] = sess.run([self._probability, self.final_state], feed)
      output = id_to_char[weighted_pick(prob)]
      res += output
      char = output
    return res

  def assign_lr(self, session, lr_value):
    session.run(tf.assign(self.lr, lr_value))

  @property
  def input_data(self):
    return self._input_data

  @property
  def targets(self):
    return self._targets

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def final_state(self):
    return self._final_state

  @property
  def stage(self):
    return self._stage

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op

  @property
  def num_steps(self):
    return self._num_steps

  @property
  def batch_size(self):
    return self._batch_size

  @property
  def misclass(self):
    return self._misclass

