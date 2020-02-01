# -*- coding: utf-8 -*-

import os
import numpy as np
import tensorflow as tf

from .BaseModel import TextClassifierBaseModel

class FastText(TextClassifierBaseModel):
    def __init__(self, num_classes, sequence_length,
                 w2v_model_embedding, vocab_size, embedding_size,
                 initializer=tf.random_normal_initializer(stddev=0.1),
                 l2_reg_lambda=0.001):
        super(FastText, self).__init__(num_classes=num_classes, sequence_length=sequence_length,
                                       w2v_model_embedding=w2v_model_embedding, vocab_size=vocab_size, embedding_size=embedding_size,
                                       initializer=tf.random_normal_initializer(stddev=0.1),
                                       l2_reg_lambda=0.001)

        self._initialize_embedding()
        self._initialize_weights()
        self.logits = self._inference()
        self.loss = self._loss()
        self.accuracy = self._accuracy()

    def _inference(self):
        sentence_embedding = tf.nn.embedding_lookup(self.Embedding, self.input_x)

        self.sentence_embedding = tf.reduce_mean(sentence_embedding, axis=1) # [None, self.embedding_size]

        logits = tf.matmul(self.sentence_embedding, self.W) + self.b
        return logits
