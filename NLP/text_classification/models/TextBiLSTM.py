# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

from .BaseModel import TextClassifierBaseModel

class TextBiLSTM(TextClassifierBaseModel):
    def __init__(self, num_classes, sequence_length,
                 w2v_model_embedding, vocab_size, embedding_size,
                 hidden_num, keep_prob,
                 initializer=tf.random_normal_initializer(stddev=0.1),
                 l2_reg_lambda=0.001):
        super(TextBiLSTM, self).__init__(num_classes=num_classes, sequence_length=sequence_length,
                                         w2v_model_embedding=w2v_model_embedding, vocab_size=vocab_size, embedding_size=embedding_size,
                                         initializer=tf.random_normal_initializer(stddev=0.1),
                                         l2_reg_lambda=0.001)

        self.hidden_num = hidden_num
        self.keep_prob = keep_prob

        self._initialize_embedding()
        self._initialize_weights()
        self.logits = self._inference()

    def _initialize_weights(self):
        with tf.name_scope('weights'):
            self.W = tf.get_variable(name='W',
                                     shape=[self.hidden_num * 2, self.num_classes],
                                     initializer=self.initializer)
            self.b = tf.get_variable(name='b', shape=[self.num_classes])

    def _inference(self):

        self.embedding_words = tf.nn.embedding_lookup(self.Embedding, self.input_x)

        rnn_drop = self._bilstm_layer()

        with tf.name_scope('output'):
            logits = tf.matmul(rnn_drop, self.W) + self.b
        
        return logits

    def _bilstm_layer(self):
        fw_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_num, state_is_tuple=True)
        bw_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_num, state_is_tuple=True)

        with tf.name_scope("dropout"):
            fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, output_keep_prob=self.keep_prob)
            bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell, output_keep_prob=self.keep_prob)

        outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell,
                                                     inputs=self.embedding_words,
                                                     dtype=tf.float32)
        outputs = tf.concat(outputs, axis=2)
        output = tf.reduce_mean(outputs, axis=1)

        return output
