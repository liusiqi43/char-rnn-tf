from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf


def _read_text(filename):
  with tf.gfile.GFile(filename, "r") as f:
    return f.read()

def _build_vocab(data):
    chars = list(set(data))
    data_size, vocab_size = len(data), len(chars)
    print('data has %d characters, %d unique.' % (data_size, vocab_size))
    char_to_id = {ch:i for i,ch in enumerate(chars)}
    return char_to_id

def raw_data(data_path):
    data = _read_text(data_path)
    char_to_id = _build_vocab(data)
    data = [char_to_id[c] for c in data]
    train_size = int(0.8 * len(data))
    print('train_size', train_size, 'test_size', len(data)-train_size)
    return {'train': data[:train_size], 'test': data[train_size:]}, char_to_id

def iterator(raw_data, batch_size, num_steps):
    """Iterate on the raw data.

    This generates batch_size pointers into the raw PTB data, and allows
    minibatch iteration along these pointers.

    Args:
        raw_data: one of the raw data outputs from ptb_raw_data.
        batch_size: int, the batch size.
        num_steps: int, the number of unrolls.

    Yields:
        Pairs of the batched data, each a matrix of shape [batch_size, num_steps].
        The second element of the tuple is the same data time-shifted to the
        right by one.

    Raises:
        ValueError: if batch_size or num_steps are too high.
    """
    raw_data = np.array(raw_data, dtype=np.int32)

    data_len = len(raw_data)
    batch_len = data_len // batch_size
    data = np.zeros([batch_size, batch_len], dtype=np.int32)
    for i in range(batch_size):
        data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

    epoch_size = (batch_len - 1) // num_steps

    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in range(epoch_size):
        x = data[:, i*num_steps:(i+1)*num_steps]
        y = data[:, i*num_steps+1:(i+1)*num_steps+1]
        yield (x, y)
