from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from glob import glob

import time

import numpy as np
import tensorflow as tf
from tensorflow.models.rnn import rnn

import reader

logging = tf.logging

tf.flags.DEFINE_float('dropout_keep_prob', 1.0, 'Dropout keep probability.')
tf.flags.DEFINE_float('init_scale', 0.1, 'Initialization scale.')
tf.flags.DEFINE_float('learning_rate', 1.0, 'Initial LR.')
tf.flags.DEFINE_float('lr_decay', 0.5, 'LR decay.')
tf.flags.DEFINE_float('max_grad_norm', 5.0, 'Maximum gradient norm.')
tf.flags.DEFINE_integer('batch_size', 32, 'Batch Size.')
tf.flags.DEFINE_integer('embedding_dim', 150, 'Dimensionality of character embedding.')
tf.flags.DEFINE_integer('hidden_size', 500, 'Hidden size of LSTM cell.')
tf.flags.DEFINE_integer('num_layers', 2, 'Number of layers of stacked LSTM cell.')
tf.flags.DEFINE_integer('max_epoch', 4, 'Max number of training epochs before LR decay.')
tf.flags.DEFINE_integer('max_max_epoch', 13, 'Stop after max_max_epoch epochs.')
tf.flags.DEFINE_integer('num_steps', 50, 'Sequence length of RNN.')
tf.flags.DEFINE_string('txt_path', '', 'Source txt file.')

FLAGS = tf.flags.FLAGS


class SpeechGen(object):
  def __init__(self, is_training, charset_size):
    self._batch_size = FLAGS.batch_size
    self._num_steps = FLAGS.num_steps
    self._hidden_size = FLAGS.hidden_size
    self._embedding_dim = FLAGS.embedding_dim
    self._dropout_keep_prob = FLAGS.dropout_keep_prob
    self._num_layers = FLAGS.num_layers
    self._charset_size = charset_size
    self._is_training = is_training

    self._input_data = tf.placeholder(tf.int32, [self._batch_size, self._num_steps])
    self._targets = tf.placeholder(tf.int32, [self._batch_size, self._num_steps])

    lstm_cell_h1 = tf.nn.rnn_cell.LSTMCell(self._hidden_size, self._embedding_dim)
    lstm_cell_h2 = tf.nn.rnn_cell.LSTMCell(self._hidden_size, self._hidden_size)
    if is_training and self._dropout_keep_prob < 1:
        lstm_cell_h1 = tf.nn.rnn_cell.DropoutWrapper(
            lstm_cell_h1, output_keep_prob=self._dropout_keep_prob)
        lstm_cell_h2 = tf.nn.rnn_cell.DropoutWrapper(
            lstm_cell_h2, output_keep_prob=self._dropout_keep_prob)
    cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_h1] + [lstm_cell_h2] * self._num_layers)

    self._initial_state = cell.zero_state(self._batch_size, tf.float32)

    with tf.device('/cpu:0'):
      embedding = tf.get_variable('embedding', [self._charset_size, self._embedding_dim])
      inputs = tf.nn.embedding_lookup(embedding, self._input_data)
      self._inputs_shape = tf.shape(inputs)

    inputs = [tf.squeeze(input_, [1])
              for input_ in tf.split(1, self._num_steps, inputs)]
    outputs, state = rnn.rnn(cell, inputs, initial_state=self._initial_state)

    output = tf.reshape(tf.concat(1, outputs), [-1, self._hidden_size])
    self._output_shape = tf.shape(output)
    softmax_w = tf.get_variable('softmax_w', [self._hidden_size, self._charset_size])
    softmax_b = tf.get_variable('softmax_b', [self._charset_size])
    logits = tf.matmul(output, softmax_w) + softmax_b
    loss = tf.nn.seq2seq.sequence_loss_by_example(
        [logits],
        [tf.reshape(self._targets, [-1])],
        [tf.ones([self._batch_size * self._num_steps])], self._charset_size)
    self._cost = cost = tf.reduce_sum(loss) / self._batch_size
    self._final_state = state

    pred = tf.argmax(logits, 1)
    labels = tf.cast(tf.reshape(self._targets, [-1]), tf.int64)
    self._misclass = 1 - tf.reduce_mean(tf.cast(tf.equal(pred, labels), tf.float32))

    if not is_training:
      return

    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                      FLAGS.max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(self.lr)
    self._train_op = optimizer.apply_gradients(zip(grads, tvars))

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
  def is_training(self):
    return self._is_training

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

def run_epoch(session, m, data, eval_op, verbose=False):
  '''Runs the model on the given data.'''
  epoch_size = ((len(data) // m.batch_size) - 1) // m.num_steps
  start_time = time.time()
  costs = 0.0
  iters = 0
  state = m.initial_state.eval()
  misclass_ = []
  for step, (x, y) in enumerate(reader.iterator(data, m.batch_size, m.num_steps)):
      cost, state, misclass, _ = session.run([m.cost, m.final_state, m.misclass, eval_op],
                                             {m.input_data: x,
                                              m.targets: y,
                                              m.initial_state: state})
      costs += cost
      iters += m.num_steps

      if verbose and step % (epoch_size // 10) == 10:
          print('[%s] %.3f perplexity: %.3f misclass:%.3f speed: %.0f wps' %
                ('train' if m.is_training else 'test', step * 1.0 / epoch_size,
                 np.exp(costs / iters), misclass,
                 iters * m.batch_size / (time.time() - start_time)))
      misclass_.append(misclass)
  return np.exp(costs / iters), np.mean(misclass_)

def main(unused_args):
    data, char_to_id = reader.raw_data(FLAGS.txt_path)

    best_misclass = 1.0
    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_uniform_initializer(-FLAGS.init_scale, FLAGS.init_scale)
        with tf.variable_scope('model', reuse=None, initializer=initializer):
            m = SpeechGen(True, len(char_to_id))
        with tf.variable_scope('model', reuse=True, initializer=initializer):
            mtest = SpeechGen(False, len(char_to_id))

        tf.initialize_all_variables().run()

        saver = tf.train.Saver()
        for i in range(FLAGS.max_max_epoch):
            lr_decay = FLAGS.lr_decay ** max(i - FLAGS.max_epoch, 0.0)
            m.assign_lr(session, FLAGS.learning_rate * lr_decay)

            print('Epoch: %d Learning rate: %.3f' % (i + 1, session.run(m.lr)))
            train_perplexity, _ = run_epoch(session, m, data['train'], m.train_op, verbose=True)
            _, misclass = run_epoch(session, mtest, data['test'], tf.no_op(), verbose=True)
            if misclass < best_misclass:
                best_misclass = misclass
                fname = 'checkpoints/obama_speech_' + str(best_misclass)
                saver.save(session, fname, global_step=i)
                print('saving', fname)

if __name__ == '__main__':
    tf.app.run()
    print('\nParameters:')
    for attr, value in sorted(FLAGS.__flags.iteritems()):
        print('{}={}'.format(attr.upper(), value))
    print('')
