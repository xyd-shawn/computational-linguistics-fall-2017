# -*- coding: utf-8 -*-

import os
import sys

import gensim
import pickle
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from gensim.models.word2vec import LineSentence
from collections import Counter

from models import poemRNN


class poemGenerator(object):
    def __init__(self, **kwargs):
        self.poem_length = kwargs.get('poem_length', 24)    # 包括标点
        self.poems_file = kwargs.get('poems_file', '../corpus/extracted_poems.txt')
        self.w2v_file = kwargs.get('w2v_file', '../corpus/word2vec.model')
        self.embed_size = kwargs.get('embed_size', 256)
        self.min_count = kwargs.get('min_count', 3)
        self.model_dir = kwargs.get('model_dir', '../tmp/')

    def process_poems(self):
        if os.path.exists(self.w2v_file):
            w2v = gensim.models.Word2Vec.load(self.w2v_file)
        else:
            w2v = gensim.models.Word2Vec(LineSentence(self.poems_file),
                                    size=self.embed_size, window=5, min_count=self.min_count)
            w2v.save(self.w2v_file)
        with open(self.poems_file, 'r') as f:
            texts = f.readlines()
        poems = []
        words = []
        for text in texts:
            temp = text.strip().split(' ')
            if len(temp) == self.poem_length + 2:
                poems.append(temp)
                words += [ww for ww in temp]
        words_count_file = '../corpus/words_count_' + str(self.poem_length) + '.pkl'
        if os.path.exists(words_count_file):
            with open(words_count_file, 'rb') as f:
                words_count = pickle.load(f)
        else:
             dic = Counter(words)
             words_count = sorted(dic.items(), key=lambda x: x[1], reverse=True)
             with open(words_count_file, 'wb') as f:
                 pickle.dump(words_count, f)
        words_index_file = '../corpus/words_index_' + str(self.poem_length) + '.pkl'
        words_map_file = '../corpus/words_map_' + str(self.poem_length) + '.pkl'
        if os.path.exists(words_index_file):
            with open(words_index_file, 'rb') as f:
                self.words_index = pickle.load(f)
            with open(words_map_file, 'rb') as f:
                self.words_map = pickle.load(f)
        else:
            i = 1
            self.words_index, self.words_map = {}, {}
            for ww, cc in words_count:
                if cc < self.min_count:
                    break
                else:
                    self.words_index[ww] = i
                    self.words_map[i] = ww
                    i += 1
            with open(words_index_file, 'wb') as f:
                pickle.dump(self.words_index, f)
            with open(words_map_file, 'wb') as f:
                pickle.dump(self.words_map, f)
        print('>>> create id2vec')
        id2vec = {}
        for i in self.words_map.keys():
            id2vec[i] = w2v[self.words_map[i]]
        id2vec[0] = np.zeros(self.embed_size)
        vocab_size = len(self.words_index)
        print('>>> create poems vector')
        poems_vec = []
        for poem in poems:
            poems_vec.append([self.words_index.get(ww, 0) for ww in poem])
        return id2vec, vocab_size, poems_vec

    def train(self, epochs):
        self.id2vec, vocab_size, poems_vec = self.process_poems()
        self.clf = poemRNN(vocab_size=vocab_size, epochs=epochs, model_dir=self.model_dir)
        self.clf.fit(poems_vec, id2vec=self.id2vec)

    def generate_poem(self, first_word):
        dd = self.words_index[first_word]
        sent = [dd]
        saver = tf.train.Saver(tf.global_variables())
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        with tf.Session() as sess:
            sess.run(init_op)
            ckpt = tf.train.latest_checkpoint(self.model_dir)
            saver.restore(sess, ckpt)
            testy, lstate = sess.run([self.clf.model_rnn['pred'], self.clf.model_rnn['last_state']],
                                    feed_dict={self.clf.model_rnn['batch_size']: 1,
                                            self.clf.model_rnn['inputs']: np.array([[self.id2vec[self.words_index['<BOS>']]]])})
            for i in range(1, self.poem_length):
                testy, lstate = sess.run([self.clf.model_rnn['pred'], self.clf.model_rnn['last_state']],
                                        feed_dict={self.clf.model_rnn['batch_size']: 1,
                                                self.clf.model_rnn['inputs']: np.array([[self.id2vec[dd]]]),
                                                self.clf.model_rnn['initial_state']: lstate})
                dd = testy[0]
                sent.append(dd)
        ss = []
        for i in range(self.poem_length):
            if sent[i] == 0:
                ss.append(' ')
            else:
                ss.append(self.words_map[sent[i]])
        print(''.join(ss))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='options')
    parser.add_argument('-m', '--modeldir', default='../tmp/', help='checkpoint path')
    parser.add_argument('-e', '--epochs', type=int, default=50, help='training epochs')
    parser.add_argument('-l', '--poemlen', type=int, default=24, help='length of poem')
    args = parser.parse_args()
    model_dir = args.modeldir
    epochs = args.epochs
    poem_length = args.poemlen
    pg = poemGenerator(model_dir=model_dir, poem_length=poem_length)
    pg.train(epochs)
    while True:
        input_first_word = input('>>> ')
        if input_first_word == 'exit':
            sys.exit()
        else:
            pg.generate_poem(input_first_word)
