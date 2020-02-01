# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

class TextRNN(object):
    def __init__(self, num_classes, sequence_length,
                 w2v_model_embedding, vocab_size, embedding_size,
                 hidden_num, hidden_size,
                 initializer=tf.random_normal_initializer(stddev=0.1),
                 l2_reg_lambda=0.001):
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.w2v_model_embedding = tf.cast(w2v_model_embedding, tf.float32)
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_num = hidden_num
        self.hidden_size = hidden_size
        self.initializer = initializer
        self.l2_reg_lambda = l2_reg_lambda
        self.l2_loss = tf.constant(0.0)

        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name='input_x')
        self.input_y = tf.placeholder(tf.int32, [None, self.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.seq_length = tf.placeholder(tf.int32, [None], name='seq_length')

        with tf.name_scope('embedding'):
            if self.w2v_model_embedding is None:
                self.Embedding = tf.get_variable(name='embedding',
                                                 shape=[self.vocab_size, self.embedding_size],
                                                 initializer=self.initializer)
            else:
                self.Embedding = tf.get_variable(name='embedding',
                                                 initializer=self.w2v_model_embedding)
        self.embedding_words = tf.nn.embedding_lookup(self.Embedding, self.input_x)

        cells = []
        for _ in range(self.hidden_size):
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_num, state_is_tuple=True)
            lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=self.keep_prob)
            cells.append(lstm_cell)
        cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)

        self.embedding_words = tf.nn.dropout(self.embedding_words, self.keep_prob)
        outputs, states = tf.nn.dynamic_rnn(cell,
                                            inputs=self.embedding_words,
                                            sequence_length=self.seq_length,
                                            dtype=tf.float32)
        