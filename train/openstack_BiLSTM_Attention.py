# -*- coding: utf-8 -*-
'''
2019-7-29 13:00 by @saulzhang
训练主文件
'''
from keras.layers import Conv1D, MaxPooling1D, Embedding,Dropout, LSTM, GRU, Bidirectional, TimeDistributed,Bidirectional
from keras.layers import Activation
from keras.models import Model,Sequential
from keras.layers.core import Lambda,RepeatVector
from keras.layers import Dense, Input, Flatten,Permute,Reshape
from keras.callbacks import EarlyStopping
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers.merge import concatenate
from keras.layers import Convolution1D, MaxPool1D, GlobalAveragePooling1D
from keras.layers import  BatchNormalization
from keras import initializers
from keras import constraints
from keras import regularizers
from keras.engine.topology import Layer
from keras.models import load_model
from keras.callbacks import Callback
from sklearn.metrics import f1_score, precision_score, recall_score
from keras import backend as K
from sklearn import metrics
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import numpy
import random as rand
import pandas as pd
import numpy as np
import os
import time
import keras
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import pickle
import logging


seed = 7
numpy.random.seed(seed)

print("设置参数")

#获取数据参数
logpath='./model_openstack/mylog.txt' #日志记录地址
modelpath='./model_openstack/' #模型保存目录

#模型训练参数
batch_size = 1024
embedding_dims = 150
epochs = 150

x_train_dataset_pkl = '../data_openstack/x_train_dataset.pkl'
y_train_dataset_pkl = '../data_openstack/y_train_dataset.pkl'
x_test_dataset_pkl = '../data_openstack/x_test_dataset.pkl'
y_test_dataset_pkl = '../data_openstack/y_test_dataset.pkl'
other_info_pkl = '../data_openstack/other_info.pkl'

print("获取数据")

x_train = pickle.load(open(x_train_dataset_pkl, 'rb'))
y_train = pickle.load(open(y_train_dataset_pkl, 'rb'))
x_test = pickle.load(open(x_test_dataset_pkl, 'rb'))
y_test = pickle.load(open(y_test_dataset_pkl, 'rb'))
maxlen = pickle.load(open(other_info_pkl, 'rb'))[0]
max_token = pickle.load(open(other_info_pkl, 'rb'))[1]
embedding_matrix = pickle.load(open(other_info_pkl, 'rb'))[2]

modelname = "LSTMAttention_预处理_Focal_loss_fixed_平衡采样1_10_"

# print("embedding_matrix.shape:",embedding_matrix.shape)
best_f1_score = 0
best_precision = 0
best_recall = 0
best_auc = 0
# word_2_idx = pickle.load(open('../data_openstack/word2idx_dict.pkl', 'rb'))


print(type(x_train))
print(type(y_train))


x_train =  np.concatenate((x_train,x_test),axis=0)
y_train = y_train + y_test

X = x_train
Y = y_train

Y = np.array(Y)

print(x_train.shape)
print(len(y_train))
print(x_test.shape)
print(len(y_test))


# Compatible with tensorflow backend
class AttLayer(Layer):
    def __init__(self, init='glorot_uniform', kernel_regularizer=None, 
                 bias_regularizer=None, kernel_constraint=None, 
                 bias_constraint=None,  **kwargs):
        self.supports_masking = True
        self.init = initializers.get(init)
        self.kernel_initializer = initializers.get(init)
        
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(kernel_regularizer)
        
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        
        super(AttLayer, self).__init__(** kwargs)

    def build(self, input_shape):
        assert len(input_shape)==3
        self.W = self.add_weight((input_shape[-1], 1),
                                 initializer=self.kernel_initializer,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint)
        self.b = self.add_weight((input_shape[1],),
                                 initializer='zero',
                                 name='{}_b'.format(self.name),
                                 regularizer=self.bias_regularizer,
                                 constraint=self.bias_constraint)
        self.u = self.add_weight((input_shape[1],),
                                 initializer=self.kernel_initializer,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint)
        
        self.built = True
        
    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        uit = K.dot(x, self.W) # (x, 40, 1)
        uit = K.squeeze(uit, -1) # (x, 40)
        uit = uit + self.b # (x, 40) + (40,)
        uit = K.tanh(uit) # (x, 40)

        ait = uit * self.u # (x, 40) * (40, 1) => (x, 1)
        ait = K.exp(ait) # (X, 1)

        if mask is not None:
            mask = K.cast(mask, K.floatx()) #(x, 40)
            ait = mask*ait #(x, 40) * (x, 40, )

        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)
        return output
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

def boolMap(arr):
    if arr > 0.4:
        return 1
    else:
        return 0

class Metrics(Callback):
    def __init__(self, filepath):
        self.file_path = filepath

    def on_train_begin(self, logs=None):
        self.val_f1s = []
        self.best_val_f1 = 0
        self.val_recalls = []
        self.val_precisions = []
        self.val_aucs = []

    def on_epoch_end(self, epoch, logs=None):
        val_predict = list(map(boolMap, self.model.predict(self.validation_data[0])))
        # val_predict = np.argmax(self.model.predict(self.validation_data[0]),axis=1)
        # confidence = np.max(self.model.predict(self.validation_data[0]),axis=1)
        # val_targ = np.argmax(self.validation_data[1],axis=1)
        val_targ = self.validation_data[1]
        # val_predict = list(map(boolMap,confidence))

        auc = roc_auc_score(val_targ, val_predict,average=None)
        _val_f1 = f1_score(val_targ, val_predict,average=None)
        _val_recall = recall_score(val_targ, val_predict,average=None)
        _val_precision = precision_score(val_targ, val_predict,average=None)
        
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        self.val_aucs.append(auc)

        cnt_1 = 0
        cnt_0 = 0

        for i in val_predict:
            if i==1:
                cnt_1+=1
            else:
                cnt_0+=1

        print("\nThe number of the example which was predicted to 0: ",cnt_0)
        print("The number of the example which was predicted to 1: ",cnt_1)
        
        print(auc, _val_f1, _val_precision, _val_recall)

        if _val_f1[1] > self.best_val_f1:
            self.model.save_weights(self.file_path, overwrite=True)
            self.best_val_f1 = _val_f1[1]
            print("best f1: {}".format(self.best_val_f1))
            global best_precision
            best_precision = _val_precision[1]
            global best_recall 
            best_recall = _val_recall[1]
            global best_auc
            best_auc = auc
        else:
            print("val f1: {}, but the best f1 is {}".format(_val_f1[1],self.best_val_f1))
        
        global best_f1_score

        best_f1_score = self.best_val_f1


        return

#灵敏度越高，漏诊率越低
def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

#特异性越高，误诊率越低
def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())
#Reference:  https://blog.csdn.net/weixin_40755306/article/details/82290150

margin = 0.8
theta = lambda t: (K.sign(t)+1.)/2.
def variant_crossentropy_loss():
    def variant_crossentropy_loss_fixed(y_true, y_pred):
        return  - theta(margin - y_pred) * y_true * K.log(y_pred + 1e-9) - theta(y_pred - 1 + margin) * (1 - y_true) * K.log(1 - y_pred + 1e-9)
    return variant_crossentropy_loss_fixed
#Reference:https://zhuanlan.zhihu.com/p/67650069?utm_source=com.tencent.tim&utm_medium=social&utm_oi=759537483925979136

def variant_focal_loss(gamma=2., alpha=0.5, rescale = False):

    gamma = float(gamma)
    alpha = float(alpha)

    def focal_loss_fixed(y_true, y_pred):
        """
        Focal loss for bianry-classification
        FL(p_t)=-rescaled_factor*alpha_t*(1-p_t)^{gamma}log(p_t)
        
        Notice: 
        y_pred is probability after sigmoid

        Arguments:
            y_true {tensor} -- groud truth label, shape of [batch_size, 1]
            y_pred {tensor} -- predicted label, shape of [batch_size, 1]

        Keyword Arguments:
            gamma {float} -- (default: {2.0})  
            alpha {float} -- (default: {0.5})

        Returns:
            [tensor] -- loss.
        """
        epsilon = 1.e-9  
        y_true = tf.convert_to_tensor(y_true, tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, tf.float32)
        model_out = tf.clip_by_value(y_pred, epsilon, 1.-epsilon)  # to advoid numeric underflow
        
        # compute cross entropy ce = ce_0 + ce_1 = - (1-y)*log(1-y_hat) - y*log(y_hat)
        ce_0 = tf.multiply(tf.subtract(1., y_true), -tf.log(tf.subtract(1., model_out)))
        ce_1 = tf.multiply(y_true, -tf.log(model_out))

        # compute focal loss fl = fl_0 + fl_1
        # obviously fl < ce because of the down-weighting, we can fix it by rescaling
        # fl_0 = -(1-y_true)*(1-alpha)*((y_hat)^gamma)*log(1-y_hat) = (1-alpha)*((y_hat)^gamma)*ce_0
        fl_0 = tf.multiply(tf.pow(model_out, gamma), ce_0)
        fl_0 = tf.multiply(1.-alpha, fl_0)
        # fl_1= -y_true*alpha*((1-y_hat)^gamma)*log(y_hat) = alpha*((1-y_hat)^gamma*ce_1
        fl_1 = tf.multiply(tf.pow(tf.subtract(1., model_out), gamma), ce_1)
        fl_1 = tf.multiply(alpha, fl_1)
        fl = tf.add(fl_0, fl_1)
        f1_avg = tf.reduce_mean(fl)
        
        if rescale:
            # rescale f1 to keep the quantity as ce
            ce = tf.add(ce_0, ce_1)
            ce_avg = tf.reduce_mean(ce)
            rescaled_factor = tf.divide(ce_avg, f1_avg + epsilon)
            f1_avg = tf.multiply(rescaled_factor, f1_avg)
        
        return f1_avg
    
    return focal_loss_fixed

def training(x_train, y_train,maxlen,max_token,embedding_matrix,embedding_dims,batch_size,epochs,logpath,modelpath,modelname):
    
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    f1_score_list = []
    auc_list = []
    precision_list = []
    recall_list = []
    cnt = 0
    f = open("result.txt",'a')
    f.write(modelname+":\n")
    for train, test in kfold.split(x_train, y_train):
        cnt += 1
        print("train=",x_train[train])
        print("train=",y_train[train])
        embedding_layer = Embedding(max_token+1,
                                output_dim=embedding_dims,
                                input_length=maxlen,
                                weights=[embedding_matrix],
                                trainable=True)

        sentence_input = Input(shape=(maxlen,), dtype='float64')
        sentence_embedding = embedding_layer(sentence_input)
        # embedded_sequences2 = embedding_layer2(sentence_input)
        # embedded_sequences = keras.layers.Add()([embedded_sequences1, embedded_sequences2])
        # print("embedded_sequences:",embedded_sequences)

        embedded_sequences = Dropout(0.25)(sentence_embedding)
        # print("embedded_sequences:",embedded_sequences)
        # embed = Embedding(len(vocab) + 1,300, input_length = 20)(inputs)
        lstm = Bidirectional(LSTM(256, dropout=0.2, recurrent_dropout=0.1, return_sequences=True))(embedded_sequences)
        # lstm2 = Bidirectional(LSTM(256, dropout=0.2, recurrent_dropout=0.1, return_sequences=True))(embedded_sequences)
        # lstm = keras.layers.Add()([lstm1, lstm2])
        # print("lstm:",lstm)
        attention = AttLayer()(lstm)
        # attention2 = AttLayer()(lstm)
        # attention = keras.layers.Add()([attention1, attention2])
        # print("attention:",attention)
        output = Dense(1, activation='sigmoid')(attention)
        # print("output:",output)
        model = Model(sentence_input, output)

        model.compile(optimizer='nadam', loss=variant_focal_loss(gamma=2., alpha=0.5, rescale = False), metrics=[sensitivity, specificity])#variant_crossentropy_loss() categorical_crossentropy_with_cost(cost=200) binary_crossentropy_with_cost(cost=200)'binary_crossentropy'[binary_crossentropy_with_cost(cost=200)] loss=[focal_loss(alpha=.25, gamma=2)] 'categorical_crossentropy'
        model.summary()

        metrics = Metrics(filepath=modelpath + modelname+ '.h5',)
        callback_lists=[metrics]#early_stopping,tensorboard,checkpoint,

        #利用class_weight实现数目较多类别的欠采样以及数目较少类别的重采样
        model.fit(x_train[train], y_train[train],validation_data=(X[test], Y[test]),class_weight = {0:1, 1:40},
            epochs=epochs, batch_size=batch_size, callbacks=callback_lists)#class_weight = {0:1, 1:40},
        
        os.rename(modelpath + modelname+ '.h5',modelpath + modelname+'(model_'+str(cnt)+'auc_'+str(best_auc)+'_f1_score_'+str(best_f1_score)+'precision_'+str(best_precision)+'recall_'+str(best_recall)+')'+ '.h5')
        
        print("model "+str(cnt)+":\n")

        f.write("f1 score: "+str(best_f1_score)+"        precision: "+str(best_precision)+"      recall: "+str(best_recall)+"\n")
        f1_score_list.append(best_f1_score)
        auc_list.append(best_auc)
        precision_list.append(best_precision)
        recall_list.append(best_recall)

    print(f1_score_list)
    print(auc_list)
    print(precision_list)
    print(recall_list)

    print("%.2f%% (+/- %.2f%%)" % (numpy.mean(f1_score_list), numpy.std(f1_score_list)))

    f.write(modelname+":\n")
    f.write(str(f1_score_list)+"\n")
    f.write("f1 score = "+str(numpy.mean(f1_score_list))+" std = "+str(numpy.std(f1_score_list))+"\n")

    f.write(str(auc_list)+"\n")
    f.write("auc = "+str(numpy.mean(auc_list))+" std = "+str(numpy.std(auc_list))+"\n")

    f.write(str(precision_list)+"\n")
    f.write("precision = "+str(numpy.mean(precision_list))+" std = "+str(numpy.std(precision_list))+"\n")

    f.write(str(recall_list)+"\n")
    f.write("recall = "+str(numpy.mean(recall_list))+" std = "+str(numpy.std(recall_list))+"\n")

training(X,Y,maxlen[0],max_token[0],embedding_matrix[0],embedding_dims,batch_size,epochs,logpath,modelpath,modelname)
