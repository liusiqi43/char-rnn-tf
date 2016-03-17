# char-rnn-tf
Multi-layer Recurrent Neural Networks (LSTM, RNN) for character-level language models in Python using Tensorflow. Based on [karpathy/char-rnn](https://github.com/karpathy/char-rnn) and his excellent [write-up](http://karpathy.github.io/2015/05/21/rnn-effectiveness/).

# Dependency
Tensorflow

# Sample data
Pulled Obama's speeches from [whitehouse press release](https://www.whitehouse.gov/briefing-room/speeches-and-remarks). 

# Usage
`python train.py --txt_path=obama.txt --session_path=./checkpoints` to train and `python sample.py` to try on sample text. 
