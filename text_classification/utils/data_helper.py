# -*- coding: utf-8 -*-

import os
import sys 
import pickle
import gensim
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

def read_and_process_data(data_path, w2v_model, save_path):
    train_data = []
    train_label = []
    label_map = {'Positive': 1, 'Negative': 0}
    with open(data_path, 'r', encoding='utf-8') as f:
        for index, line in enumerate(f.readlines()):
            if index == 0:
                continue
            line = line.strip().split(',')
            if len(line) != 3:
                continue
            if line[-1] not in label_map:
                continue
            s_id, content, label = line
            train_data.append(content)
            train_label.append(label_map[label])
    train_label = to_categorical(train_label, num_classes=2)

    train_x, valid_x, train_y, valid_y = train_test_split(train_data, train_label, test_size=0.15, random_state=2020)
    
    maxlen = max([len(c.split(' ')) for c in train_data])

    ## Tokenize the sentences
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_data)
    word2index = tokenizer.word_index
    print(len(word2index))
    embedding = generate_embedding(word2index, w2v_model)

    train_x = tokenizer.texts_to_sequences(train_x)
    train_x = pad_sequences(train_x, maxlen=maxlen)

    valid_x = tokenizer.texts_to_sequences(valid_x)
    valid_x = pad_sequences(valid_x, maxlen=maxlen)

    np.save(save_path + 'train_x.npy', train_x)
    np.save(save_path + 'train_y.npy', train_y)
    np.save(save_path + 'valid_x.npy', valid_x)
    np.save(save_path + 'valid_y.npy', valid_y)
    np.save(save_path + 'embedding.npy', embedding)

    pickle.dump(word2index, open(save_path + 'word2index.pkl', 'wb'))

    print('vocab size: {}'.format(len(word2index) + 1))

    # maxlen = train_x.shape[1]
    # vocab_size = len(word2index) + 1
    # index2word = {v: k for k, v in word2index.items()}

def generate_embedding(word2index, w2v_model):
    embedding = np.zeros((len(word2index) + 1, 200))
    for word, index in word2index.items():
        try:
            embedding[index] = w2v_model[word]
        except:
            continue
    return embedding

def load_w2v_model(w2v_model_path):
    return gensim.models.KeyedVectors.load_word2vec_format(w2v_model_path, binary=False)

def data_process(content):
    pass

def filter_stop_words(content, stop_words):
    pass
    
def next_batch(train_x, train_y, batch_size, shuffle=True):
    data_size = len(train_x)
    num_batches_per_epoch = int(data_size / batch_size) + 1
    # while True:
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = train_x[shuffle_indices]
        shuffled_data_y = train_y[shuffle_indices]
    else:
        shuffled_data, shuffled_data_y = train_x, train_y
        
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)

        yield shuffled_data[start_index:end_index], shuffled_data_y[start_index:end_index]

def load_data(data_path):
    train_x = np.load(data_path + 'train_x.npy')
    train_y = np.load(data_path + 'train_y.npy')
    valid_x = np.load(data_path + 'valid_x.npy')
    valid_y = np.load(data_path + 'valid_y.npy')
    embedding = np.load(data_path + 'embedding.npy')

    word2index = pickle.load(open(data_path + 'word2index.pkl', 'rb'))
    index2word = {v: k for k, v in word2index.items()}
    vocab_size = len(word2index) + 1
    maxlen = len(train_x[0])

    return train_x, train_y, valid_x, valid_y, embedding, word2index, index2word, vocab_size, maxlen

if __name__ == "__main__":
    data_path = '../text_data/raw_data/train.csv'
    w2v_model_path = '../text_data/w2v_model/text_w2v_model.txt'
    w2v_model = load_w2v_model(w2v_model_path)
    read_and_process_data(data_path, w2v_model, '../text_data/input_data/')