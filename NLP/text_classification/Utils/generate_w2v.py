# -*- coding: utf-8 -*-

import os
import sys
import time
import numpy as np
import gensim

def load_data(data_path):
    train_data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip().split(',')
            s_id, content, label = line
            train_data.append(content.strip().split())
    return train_data

def train_w2v(train_data, model_path):
    start_time = time.time()
    model = gensim.models.Word2Vec(train_data, size=200, window=5, min_count=5, workers=3, iter=10)
    print('train done, time used {:.4f} min.'.format((time.time() - start_time) / 60))
    model.save_word2vec_format(model_path, binary=False)

if __name__ == "__main__":
    train_data = load_data('../text_data/raw_data/train.csv')
    print(len(train_data))
    print(train_data[:3])

    train_w2v(train_data, '../text_data//w2v_model/text_w2v_model.txt')