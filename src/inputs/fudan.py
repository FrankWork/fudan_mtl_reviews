import os
import re
from collections import namedtuple
import tensorflow as tf

from inputs import util

DATA_DIR = "data/mtl-dataset"
DATASETS = ['apparel', 'baby', 'books', 'camera_photo',  'electronics', 
      'health_personal_care', 'imdb', 'kitchen_housewares', 'magazines', 
      'music', 'software', 'sports_outdoors', 'toys_games', 'video']
# 'dvd','MR',
SUFFIX = ['.task.train', '.task.test', '.task.unlabel']
Raw_Example = namedtuple('Raw_Example', 'label task sentence')
MTL_VOCAB_FILE = "data/generated/vocab.mtl.txt"
OUT_DIR = "data/generated"
MAX_LEN = 500

def get_task_name(task_id):
  return DATASETS[task_id]

def _load_raw_data_from_file(filename, task_id):
  data = []
  with open(filename) as f:
    # try:
    for line in f:
      segments = line.strip().split('\t')
      if len(segments) == 2:
        label = int(segments[0])
        tokens = segments[1].split(' ')
        example = Raw_Example(label, task_id, tokens)
        data.append(example)
    # except UnicodeDecodeError:
    #   print(filename)
    #   exit()
  return data

def _load_raw_data(dataset_name, task_id):
  train_file = os.path.join(DATA_DIR, dataset_name+'.task.train')
  train_data = _load_raw_data_from_file(train_file, task_id)
  test_file = os.path.join(DATA_DIR, dataset_name+'.task.test')
  test_data = _load_raw_data_from_file(test_file, task_id)
  return train_data, test_data

def load_raw_data():
  for task_id, dataset in enumerate(DATASETS):
    yield _load_raw_data(dataset, task_id)

def build_vocab(raw_data):
  vocab = set()
  for example in raw_data:
    for w in example.sentence:
        vocab.add(w)

  return vocab

def _build_sequence_example(raw_example):
  '''build tf.train.SequenceExample from Raw_Example
  context features : label, task
  sequence features: sentence

  Args: 
    raw_example : type Raw_Example._asdict()

  Returns:
    tf.trian.SequenceExample
  '''
  ex = tf.train.SequenceExample()

  label = raw_example['label']
  ex.context.feature['label'].int64_list.value.append(label)

  task = raw_example['task']
  ex.context.feature['task'].int64_list.value.append(task)

  for word_id in raw_example['sentence']:
    word = ex.feature_lists.feature_list['sentence'].feature.add()
    word.int64_list.value.append(word_id)
  
  return ex

def write_as_tfrecord(train_data, test_data, task_id, vocab2id):
  '''convert the raw data to TFRecord format and write to disk
  '''
  dataset = DATASETS[task_id]
  train_record_file = os.path.join(OUT_DIR, dataset+'.train.tfrecord')
  test_record_file = os.path.join(OUT_DIR, dataset+'.test.tfrecord')

  util.write_as_tfrecord(train_data, 
                         vocab2id, 
                         train_record_file, 
                         MAX_LEN, 
                         _build_sequence_example)
  util.write_as_tfrecord(test_data, 
                         vocab2id, 
                         test_record_file, 
                         MAX_LEN, 
                         _build_sequence_example)

  util._shuf_and_write(train_record_file)

def _parse_tfexample(serialized_example):
  '''parse serialized tf.train.SequenceExample to tensors
  context features : label, task
  sequence features: sentence
  '''
  context_features={'label'    : tf.FixedLenFeature([], tf.int64),
                    'task'    : tf.FixedLenFeature([], tf.int64)}
  sequence_features={'sentence': tf.FixedLenSequenceFeature([], tf.int64)}
  context_dict, sequence_dict = tf.parse_single_sequence_example(
                      serialized_example,
                      context_features   = context_features,
                      sequence_features  = sequence_features)

  sentence = sequence_dict['sentence']
  label = context_dict['label']
  task = context_dict['task']

  return task, label, sentence

def read_tfrecord(epoch, batch_size):
  for dataset in DATASETS:
    train_record_file = os.path.join(OUT_DIR, dataset+'.train.tfrecord')
    test_record_file = os.path.join(OUT_DIR, dataset+'.test.tfrecord')

    train_data = util.read_tfrecord(train_record_file, 
                                    epoch, 
                                    batch_size, 
                                    _parse_tfexample, 
                                    shuffle=True)

    test_data = util.read_tfrecord(test_record_file, 
                                    epoch, 
                                    400, 
                                    _parse_tfexample, 
                                    shuffle=False)
    yield train_data, test_data
