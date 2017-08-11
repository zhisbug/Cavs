import numpy as np
import os

data_dir = '/users/hzhang2/projects/Cavs/apps/lstm/sst'
splits = ['train', 'test', 'dev']


class Vocab(object):
  def __init__(self, path):
    self.words = []
    self.word2idx = {}
    self.idx2word = {}
    self.load(path)

  def load(self, path):
    with open(path, 'r') as f:
      for line in f:
        w = line.strip()
        assert w not in self.words
        self.words.append(w)
        self.word2idx[w] = len(self.words) - 1
        self.idx2word[self.word2idx[w]] = w

  def encode(self, word):
    return self.word2idx[word]

  def decode(self, idx):
    return self.idx2word[idx]

print('Build vocabulary...')
vocab_path = os.path.join(data_dir, 'vocab-cased.txt')
vocab = Vocab(vocab_path)

def transform(sentence):
  sentence = [w for w in sentence.strip().split()]
  indices = []
  for w in sentence:
    idx = vocab.encode(w)
    assert(idx >= 0)
    indices.append(idx)
  assert(len(indices) == len(sentence))
  return indices, len(sentence)


max_length = 0
max_graph_length = 0
print('Transform from sentences to indices...')
for split in splits:
  word_path = os.path.join(data_dir, split, 'sents.txt') 
  label_path = os.path.join(data_dir, split, 'labels.txt') 
  graph_path = os.path.join(data_dir, split, 'parents.txt')

  indices_path = os.path.join(data_dir, split, 'sents_idx.txt')

  with open(word_path, 'r') as wordfile, open(indices_path, 'w') as indices_file:
    while True:
      sentence = wordfile.readline()
      if not sentence:
        break
      indices, length = transform(sentence)
      if length > max_length:
        max_length = length
      # write the indices to a new file
      for i in range(len(indices)):
        indices_file.write('%d' % (indices[i]))
        if i < len(indices)-1 :
          indices_file.write(' ')
        else:
          indices_file.write('\n')



  with open(graph_path, 'r') as graphfile:
    while True:
      graph = graphfile.readline()
      if not graph:
        break
      parents = [w for w in graph.strip().split()]
      graph_len = len(parents)
      if graph_len > max_graph_length:
        max_graph_length = graph_len

print('Done...Max sentence length: %d' % max_length)
print('Done...Max graph length: %d' % max_graph_length)
