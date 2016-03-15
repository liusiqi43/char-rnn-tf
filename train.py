from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cPickle as pickle
import numpy as np
import os
import reader
import tensorflow as tf
import time

from models import CharRNN, Config

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
tf.flags.DEFINE_string('session_dir', 'checkpoints', 'Trained session dir to load.')

FLAGS = tf.flags.FLAGS

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
            (m.stage, step * 1.0 / epoch_size, np.exp(costs / iters), misclass,
             iters * m.batch_size / (time.time() - start_time)))
    misclass_.append(misclass)
  return np.exp(costs / iters), np.mean(misclass_)


def main(unused_args):
  data, char_to_id = reader.raw_data(FLAGS.txt_path)
  config = Config(batch_size = FLAGS.batch_size, num_steps = FLAGS.num_steps,
                  hidden_size = FLAGS.hidden_size,
                  embedding_dim = FLAGS.embedding_dim,
                  dropout_keep_prob = FLAGS.dropout_keep_prob,
                  num_layers = FLAGS.num_layers,
                  max_grad_norm = FLAGS.max_grad_norm,
                  charset_size = len(char_to_id))
  with tf.Graph().as_default(), tf.Session() as session:
    initializer = tf.random_uniform_initializer(-FLAGS.init_scale,
                          FLAGS.init_scale)
    with tf.variable_scope('model', reuse=None, initializer=initializer):
      m = CharRNN('train', config)
    with tf.variable_scope('model', reuse=True, initializer=initializer):
      mtest = CharRNN('test', config)

    tf.initialize_all_variables().run()
    saver = tf.train.Saver()
    best_misclass = 1.0
    with open(os.path.join(FLAGS.session_dir, 'labels.pkl'), 'wb') as f:
      pickle.dump(char_to_id, f)
    with open(os.path.join(FLAGS.session_dir, 'config.pkl'), 'wb') as f:
      pickle.dump(config, f)
    for i in range(FLAGS.max_max_epoch):
      lr_decay = FLAGS.lr_decay ** max(i - FLAGS.max_epoch, 0.0)
      m.assign_lr(session, FLAGS.learning_rate * lr_decay)

      print('Epoch: %d Learning rate: %.3f' % (i + 1,
                           session.run(m.lr)))
      train_perplexity, _ = run_epoch(session, m, data['train'], m.train_op, verbose=True)
      _, misclass = run_epoch(session, mtest, data['test'], tf.no_op(), verbose=True)
      if misclass < best_misclass:
        best_misclass = misclass
        fname = os.path.join(FLAGS.session_dir,
                             'obama_speech_' + str(best_misclass))
        saver.save(session, fname, global_step=i)
        print(fname, 'checkpoint saved')

if __name__ == '__main__':
  tf.app.run()
  print('\nParameters:')
  for attr, value in sorted(FLAGS.__flags.iteritems()):
    print('{}={}'.format(attr.upper(), value))
  print('')
