from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import tensorflow as tf
import time

from models import CharRNN, Config

tf.flags.DEFINE_integer('num_steps', 100, 'Sequence length of RNN.')
tf.flags.DEFINE_string('session_dir', 'checkpoints', 'Trained session dir to load.')

FLAGS = tf.flags.FLAGS

def main(unused_args):
  with open(os.path.join(FLAGS.session_dir, 'labels.pkl')) as f:
    char_to_id = pickle.load(f)
  with open(os.path.join(FLAGS.session_dir, 'config.pkl')) as f:
    config = pickle.load(f)
  with tf.variable_scope('model'):
    m = CharRNN('infer', config)
  with tf.Session() as sess:
    tf.initialize_all_variables().run()
    saver = tf.train.Saver(tf.all_variables())
    ckpt = tf.train.get_checkpoint_state(FLAGS.session_dir)
    if ckpt and ckpt.model_checkpoint_path:
      saver.restore(sess, ckpt.model_checkpoint_path)
      print(ckpt.model_checkpoint_path, 'restored')

      while True:
        seed = raw_input('seed:')
        start_time = time.time()
        print(m.sample(sess, char_to_id, FLAGS.num_steps, seed))
        print(FLAGS.num_steps / (time.time() - start_time), 'cps')

if __name__ == '__main__':
    tf.app.run()
