# -*- coding: utf-8 -*-

import os
import sys

import copy
import numpy as np
import pandas as pd
import tensorflow as tf


class poemRNN(object):

    def __init__(self, **kwargs):
        self.model_name = kwargs.get('model_name', 'lstm')
        self.num_layers = kwargs.get('num_layers', 2)
        self.embed_size = kwargs.get('embed_size', 256)
        self.embed_layer = kwargs.get('embed_layer', False)
        self.hidden_units = kwargs.get('hidden_units', 256)
        self.vocab_size = kwargs.get('vocab_size', 2000)
        self.ckpt_steps = kwargs.get('ckpt_steps', 10)
        self.lr = kwargs.get('lr', 0.01)
        self.epochs = kwargs.get('epochs', 50)
        self.model_dir = kwargs.get('model_dir', '../tmp/')

        batch_size = tf.placeholder(tf.int32, shape=[])
        targets = tf.placeholder(tf.int32, shape=[None, None])
        if self.model_name == 'rnn':
            rnn_cell = tf.contrib.rnn.BasicRNNCell
        elif self.model_name == 'gru':
            rnn_cell = tf.contrib.rnn.GRUCell
        else:
            rnn_cell = tf.contrib.rnn.BasicLSTMCell
        cell = rnn_cell(num_units=self.hidden_units, state_is_tuple=True)
        cells = tf.contrib.rnn.MultiRNNCell([cell] * self.num_layers, state_is_tuple=True)
        initial_state = cells.zero_state(batch_size, tf.float32)
        if self.embed_layer:
            inputs = tf.placeholder(tf.int32, shape=[None, None])
            with tf.device('/cpu:0'):
                embedding = tf.get_variable('embedding',
                            initializer=tf.random_uniform([self.vocab_size + 1, self.embed_size], -1.0, 1.0))
                input_data = tf.nn.embedding_lookup(embedding, inputs)
            output, last_state = tf.nn.dynamic_rnn(cells, input_data, initial_state=initial_state)
        else:
            inputs = tf.placeholder(tf.float32, shape=[None, None, self.embed_size])
            output, last_state = tf.nn.dynamic_rnn(cells, inputs, initial_state=initial_state)
        outputs = tf.reshape(output, [-1, self.hidden_units])
        w = tf.Variable(tf.truncated_normal([self.hidden_units, self.vocab_size + 1], stddev=0.01))
        b = tf.Variable(tf.zeros(shape=[self.vocab_size + 1]))
        logits = tf.nn.bias_add(tf.matmul(outputs, w), bias=b)
        labels = tf.one_hot(tf.reshape(targets, [-1]), depth=self.vocab_size + 1)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
        pred = tf.argmax(tf.nn.softmax(logits), axis=1)
        acc = tf.metrics.accuracy(pred, tf.argmax(labels, axis=1))
        train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)
        self.model_rnn = {}
        self.model_rnn['batch_size'] = batch_size
        self.model_rnn['inputs'] = inputs
        self.model_rnn['targets'] = targets
        self.model_rnn['pred'] = pred
        self.model_rnn['loss'] = loss
        self.model_rnn['acc'] = acc
        self.model_rnn['train_op'] = train_op
        self.model_rnn['initial_state'] = initial_state
        self.model_rnn['last_state'] = last_state

    def fit(self, poems_vec, **kwargs):
        temp1= np.array(poems_vec)
        temp2 = np.array(copy.deepcopy(poems_vec))
        temp2[:, :-1] = temp1[:, 1:]
        trainy = temp2
        if not self.embed_layer:
            id2vec = kwargs.get('id2vec')
            trainx = np.zeros((temp1.shape[0], temp1.shape[1], self.embed_size))
            for i in range(temp1.shape[0]):
                for j in range(temp1.shape[1]):
                    if temp1[i, j] in id2vec.keys():
                        trainx[i, j] = id2vec[temp1[i, j]]
        else:
            trainx = temp1
        batch_size = kwargs.get('batch_size', 100)
        iters_per_epoch = trainx.shape[0] // batch_size
        saver = tf.train.Saver(tf.global_variables())
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        with tf.Session() as sess:
            sess.run(init_op)
            start_epoch = 0
            ckpt = tf.train.latest_checkpoint(self.model_dir)
            if ckpt:
                saver.restore(sess, ckpt)
                start_epoch += int(ckpt.split('-')[-1])
                print('>>> restore from checkpoint %s' % (ckpt))
            print('>>> start training')
            for i in range(start_epoch, self.epochs):
                idx = np.arange(trainx.shape[0])
                np.random.shuffle(idx)
                trainx, trainy = trainx[idx], trainy[idx]
                total_loss, total_acc = 0, 0
                for j in range(iters_per_epoch):
                    batchx = trainx[(batch_size * j):(batch_size * (j + 1))]
                    batchy = trainy[(batch_size * j):(batch_size * (j + 1))]
                    train_loss, train_acc, _ = sess.run([self.model_rnn['loss'],
                                                    self.model_rnn['acc'],
                                                    self.model_rnn['train_op']],
                                                    feed_dict={self.model_rnn['batch_size']: batch_size,
                                                            self.model_rnn['inputs']: batchx,
                                                            self.model_rnn['targets']: batchy})
                    total_loss += train_loss
                    temp1, temp2 = train_acc
                    total_acc += temp1
                print('epoch: %d, train loss: %.5f, train acc: %.5f'
                            % (i + 1, total_loss / iters_per_epoch, total_acc / iters_per_epoch))
                if (i + 1) % self.ckpt_steps == 0:
                    saver.save(sess, os.path.join(self.model_dir, 'poems'), global_step=i)
