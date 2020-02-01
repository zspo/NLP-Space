# -*- coding: utf-8 -*-

import os
import numpy as np
import tensorflow as tf

from .BaseModel import TextClassifierBaseModel

class TextCNN(TextClassifierBaseModel):
    def __init__(self, filter_sizes, num_filters, num_classes, sequence_length,
                 w2v_model_embedding, vocab_size, embedding_size,
                 keep_prob=0.5,
                 initializer=tf.random_normal_initializer(stddev=0.1),
                 l2_reg_lambda=0.001):
        super(TextCNN, self).__init__(num_classes=num_classes, sequence_length=sequence_length,
                                      w2v_model_embedding=w2v_model_embedding, vocab_size=vocab_size, embedding_size=embedding_size,
                                      initializer=tf.random_normal_initializer(stddev=0.1),
                                      l2_reg_lambda=0.001)

        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.num_filters_total = self.num_filters * len(self.filter_sizes)
        self.keep_prob = keep_prob

        self._initialize_embedding()
        self._initialize_weights()
        self.logits = self._inference()
        self.loss = self._loss()
        self.accuracy = self._accuracy()

    def _initialize_weights(self):
        with tf.name_scope('weights'):
            self.W = tf.get_variable(name='W',
                                     shape=[self.num_filters_total, self.num_classes],
                                     initializer=self.initializer)
            self.b = tf.get_variable(name='b', shape=[self.num_classes])

    def _inference(self):
        self.embedding_words = tf.nn.embedding_lookup(self.Embedding, self.input_x)  # [None, sequence_length, embedding_size]
        # [None, sequence_length, embedding_size, 1]. expand dimension so meet input requirement of 2d-conv
        self.sentence_embedding_expanded = tf.expand_dims(self.embedding_words, -1)

        conv_out = self._conv_layer()

        with tf.name_scope('output'):
            logits = tf.matmul(conv_out, self.W) + self.b
        return logits

    def _conv_layer(self):
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.variable_scope('convolution-pooling-{}'.format(i)):
                filter = tf.get_variable(name='filter-{}'.format(filter_size),
                                         shape=[filter_size, self.embedding_size, 1, self.num_filters],
                                         initializer=self.initializer,)
                # Conv.Input: given an input tensor of shape `[batch, in_height, in_width, in_channels]` and a filter / kernel tensor of shape `[filter_height, filter_width, in_channels, out_channels]`
                # Conv.Returns: A `Tensor`. Has the same type as `input`.
                conv = tf.nn.conv2d(self.sentence_embedding_expanded,
                                    filter,
                                    strides=[1, 1, 1, 1],
                                    padding='VALID',
                                    name='conv')

                b = tf.get_variable(name='b-{}'.format(filter_size), shape=[self.num_filters])
                h = tf.nn.relu(tf.nn.bias_add(conv, b), 'relu')

                pooled = tf.nn.max_pool(h,
                                        ksize=[1, self.sequence_length - filter_size + 1, 1, 1],
                                        strides=[1, 1, 1, 1],
                                        padding='VALID',
                                        name='pool')
                pooled_outputs.append(pooled)

        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flatten = tf.reshape(h_pool, [-1, self.num_filters_total])

        with tf.name_scope('dropout'):
            h_drop = tf.nn.dropout(h_pool_flatten, self.keep_prob)
        
        return h_drop
