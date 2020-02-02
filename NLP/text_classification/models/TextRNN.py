# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

from .BaseModel import TextClassifierBaseModel

class TextRNN(TextClassifierBaseModel):
    def __init__(self, num_classes, sequence_length,
                 w2v_model_embedding, vocab_size, embedding_size,
                 hidden_num, hidden_size, keep_prob,
                 initializer=tf.random_normal_initializer(stddev=0.1),
                 l2_reg_lambda=0.001):
        super(TextRNN, self).__init__(num_classes=num_classes, sequence_length=sequence_length,
                                      w2v_model_embedding=w2v_model_embedding, vocab_size=vocab_size, embedding_size=embedding_size,
                                      initializer=tf.random_normal_initializer(stddev=0.1),
                                      l2_reg_lambda=0.001)

        self.hidden_num = hidden_num
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob

        self._initialize_embedding()
        self._initialize_weights()
        self.logits = self._inference()

    def _initialize_weights(self):
        with tf.name_scope('weights'):
            self.W = tf.get_variable(name='W',
                                     shape=[self.hidden_size, self.num_classes],
                                     initializer=self.initializer)
            self.b = tf.get_variable(name='b', shape=[self.num_classes])

    def _inference(self):

        self.embedding_words = tf.nn.embedding_lookup(self.Embedding, self.input_x)

        rnn_drop = self._rnn_layer()

        with tf.name_scope('output'):
            logits = tf.matmul(rnn_drop, self.W) + self.b
        
        return logits

    def _rnn_layer(self):
        cells = []
        for _ in range(self.hidden_size):
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_num, state_is_tuple=True)
            lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=self.keep_prob)
            cells.append(lstm_cell)
        cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)

        outputs, states = tf.nn.dynamic_rnn(cell,
                                            inputs=self.embedding_words,
                                            dtype=tf.float32)
        outputs = tf.reduce_mean(outputs, axis=1)

        with tf.name_scope('dropout'):
            rnn_drop = tf.nn.dropout(outputs, self.keep_prob)
        return rnn_drop
