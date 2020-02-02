# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import time
import os
import sys
sys.path.append('..')

from Utils import data_helper
from models.FastText import FastText
from models.TextCNN import TextCNN

FLAGS = tf.app.flags.FLAGS
# Data params
tf.app.flags.DEFINE_string(
    'data_path', '../text_data/input_data/', 'input data path')
# Model params
# Model params
tf.app.flags.DEFINE_string("filter_sizes", "2,3,4",
                           "textcnn model, convolution filter sizes")
tf.app.flags.DEFINE_integer(
    "num_filters", 2, "textcnn model, convolution filter nums")
tf.app.flags.DEFINE_integer("num_classes", 2, "num_classes")
tf.app.flags.DEFINE_float(
    "keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
# Training params
tf.app.flags.DEFINE_float("learning_rate", 0.01,
                          "learning_rate (default: 0.01)")
tf.app.flags.DEFINE_integer(
    "epochs", 10, "Number of training epochs (default: 10)")
tf.app.flags.DEFINE_integer("batch_size", 512, "Batch Size (default: 64)")
tf.app.flags.DEFINE_integer("checkpoint_every", 100,
                            "Save model every steps (default: 100)")
tf.app.flags.DEFINE_string("checkpoint_dir", './model_save/', "checkpoint_dir")

train_x, train_y, valid_x, valid_y, embedding, word2index, index2word, vocab_size, maxlen = data_helper.load_data('../text_data/input_data/')
print(train_x.shape)
print(vocab_size)
print(embedding.shape)
print(embedding.dtype)
print(maxlen)


model = FastText(
    num_classes=FLAGS.num_classes,
    sequence_length=maxlen,
    w2v_model_embedding=embedding,
    vocab_size=vocab_size,
    embedding_size=200)

# model = TextCNN(filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
#                 num_filters=FLAGS.num_filters,
#                 num_classes=FLAGS.num_classes,
#                 sequence_length=maxlen,
#                 w2v_model_embedding=embedding,
#                 vocab_size=vocab_size,
#                 embedding_size=200)

optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics='accuracy')
model.fit(train_x, train_y,
          batch_size=128,
          epochs=2,
          verbose=1,
          valid_x=valid_x,
          valid_y=valid_y,)
predict_scores = model.predict(train_x)
print(predict_scores[:5])


