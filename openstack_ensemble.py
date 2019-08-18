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
from keras import backend as K
from keras.layers import Embedding, Dense, SimpleRNN, Lambda, Concatenate, Conv1D, GlobalMaxPooling1D
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
modelname = "RCNN_Attention_预处理_Focal_loss_fixed_平衡采样1_1_"
best_f1_score = 0
best_precision = 0
best_recall = 0
best_auc = 0
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
        val_predict = list(map(boolMap, self.model.predict([self.validation_data[0],self.validation_data[1],self.validation_data[2]])))
        # val_predict = np.argmax(self.model.predict(self.validation_data[0]),axis=1)
        # confidence = np.max(self.model.predict(self.validation_data[0]),axis=1)
        # val_targ = np.argmax(self.validation_data[1],axis=1)
        # print(self.validation_data[0].shape)
        # print(self.validation_data[1].shape)
        # print(self.validation_data[2].shape)
        # print("val_predict:",val_predict)
        val_targ = self.validation_data[3]
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


def CRNNAttentionModel(sentence_input,maxlen,max_token,embedding_matrix,embedding_dims):

    embedder = Embedding(max_token+1,
                            output_dim=embedding_dims,
                            input_length=maxlen,
                            weights=[embedding_matrix],
                            trainable=True)
    input_current = sentence_input[0]
    input_left = sentence_input[1]
    input_right = sentence_input[2]
    embedding_current = embedder(input_current)
    embedding_left = embedder(input_left)
    embedding_right = embedder(input_right)
    x_left = Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.1, return_sequences=True))(embedding_left)
    x_right = Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.1, return_sequences=True, go_backwards=True))(embedding_right)
    x_right = Lambda(lambda x: K.reverse(x, axes=1))(x_right)   
    x = Concatenate(axis=2)([x_left, embedding_current, x_right])
    attention = AttLayer()(x)
    att_d = Dense(64, activation='relu')(attention)
    x = Conv1D(64, kernel_size=1, activation='tanh')(x)
    x = GlobalMaxPooling1D()(x)
    x = Concatenate(axis=-1)([x,att_d])
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=[input_current, input_left, input_right], outputs=output)

    return model


def AttBilstm(sentence_input,maxlen,max_token,embedding_matrix,embedding_dims):
    embedding_layer = Embedding(max_token+1,
                            output_dim=embedding_dims,
                            input_length=maxlen,
                            weights=[embedding_matrix],
                            trainable=True)
    sentence_input = Input(shape=(maxlen,), dtype='float64')
    sentence_embedding = embedding_layer(sentence_input)
    embedded_sequences = Dropout(0.25)(sentence_embedding)
    lstm = Bidirectional(LSTM(256, dropout=0.2, recurrent_dropout=0.1, return_sequences=True))(embedded_sequences)
    attention = AttLayer()(lstm)
    output = Dense(1, activation='sigmoid')(attention)
    model = Model(sentence_input, output)
    return model


def ensemble(x_train, y_train,maxlen,max_token,embedding_matrix,embedding_dims):
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    f1_score_list = []
    auc_list = []
    precision_list = []
    recall_list = []
    cnt = 0

    for train, test in kfold.split(x_train, y_train):

        # print("train=",x_train[train])
        # print("train=",y_train[train])

        x_train_current = X[train]
        x_train_left = np.hstack([np.expand_dims(X[train][:, 0], axis=1), X[train][:, 0:-1]])
        x_train_right = np.hstack([X[train][:, 1:], np.expand_dims(X[train][:, -1], axis=1)])
        x_test_current = X[test]
        x_test_left = np.hstack([np.expand_dims(X[test][:, 0], axis=1), X[test][:, 0:-1]])
        x_test_right = np.hstack([X[test][:, 1:], np.expand_dims(X[test][:, -1], axis=1)])
        
        # print('x_train_current shape:', x_train_current.shape)
        # print('x_train_left shape:', x_train_left.shape)
        # print('x_train_right shape:', x_train_right.shape)
        # print('x_test_current shape:', x_test_current.shape)
        # print('x_test_left shape:', x_test_left.shape)
        # print('x_test_right shape:', x_test_right.shape)

        input_current= Input(shape=(maxlen,), dtype='float64')
        input_left= Input(shape=(maxlen,), dtype='float64')
        input_right= Input(shape=(maxlen,), dtype='float64')
        
        path1 = [
        'RCNN_Attention_预处理_Focal_loss_fixed_平衡采样1_1_(model_1auc_0.7077280064568201_f1_score_0.5128205128205129precision_0.6666666666666666recall_0.4166666666666667).h5',
        'RCNN_Attention_预处理_Focal_loss_fixed_平衡采样1_1_(model_2auc_0.6449858757062148_f1_score_0.3684210526315789precision_0.5recall_0.2916666666666667).h5',
        'RCNN_Attention_预处理_Focal_loss_fixed_平衡采样1_1_(model_3auc_0.6458333333333334_f1_score_0.45161290322580644precision_1.0recall_0.2916666666666667).h5',
        'RCNN_Attention_预处理_Focal_loss_fixed_平衡采样1_1_(model_4auc_0.666182405165456_f1_score_0.4444444444444444precision_0.66666667recall_0.33333333).h5',
        'RCNN_Attention_预处理_Focal_loss_fixed_平衡采样1_1_(model_5auc_0.6664245359160613_f1_score_0.47058823529411764precision_0.8recall_0.3333333333333333).h5',
        'RCNN_Attention_预处理_Focal_loss_fixed_平衡采样1_1_(model_6auc_0.6873789346246973_f1_score_0.5294117647058825precision_0.9recall_0.375).h5',
        'RCNN_Attention_预处理_Focal_loss_fixed_平衡采样1_1_(model_7auc_0.6455912025827281_f1_score_0.42424242424242425precision_0.7777777777777778recall_0.2916666666666667).h5',
        'RCNN_Attention_预处理_Focal_loss_fixed_平衡采样1_1_(model_8auc_0.666182405165456_f1_score_0.4444444444444444precision_0.6666666666666666recall_0.3333333333333333).h5',
        'RCNN_Attention_预处理_Focal_loss_fixed_平衡采样1_1_(model_9auc_0.6665455719706143_f1_score_0.48484848484848486precision_0.8888888888888888recall_0.3333333333333333).h5',
        'RCNN_Attention_预处理_Focal_loss_fixed_平衡采样1_1_(model_10auc_0.7167858308675644_f1_score_0.5263157894736841precision_0.6666666666666666recall_0.43478260869565216).h5']

        path2 = ['LSTMAttention_预处理_Focal_loss_fixed_平衡采样1_10_(model_1auc_0.7076069410815174_f1_score_0.5precision_0.625recall_0.4166666666666667).h5',
        'LSTMAttention_预处理_Focal_loss_fixed_平衡采样1_10_(model_2auc_0.6039245359160613_f1_score_0.3225806451612903precision_0.7142857142857143recall_0.20833333333333334).h5',
        'LSTMAttention_预处理_Focal_loss_fixed_平衡采样1_10_(model_3auc_0.6859261501210654_f1_score_0.3913043478260869precision_0.4090909090909091recall_0.375).h5',
        'LSTMAttention_预处理_Focal_loss_fixed_平衡采样1_10_(model_4auc_0.6453490718321228_f1_score_0.4precision_0.6363636363636364recall_0.2916666666666667).h5',
        'LSTMAttention_预处理_Focal_loss_fixed_平衡采样1_10_(model_5auc_0.6245157384987893_f1_score_0.35294117647058826precision_0.6recall_0.25).h5',
        'LSTMAttention_预处理_Focal_loss_fixed_平衡采样1_10_(model_6auc_0.7074858757062148_f1_score_0.48780487804878053precision_0.5882352941176471recall_0.4166666666666667).h5',
        'LSTMAttention_预处理_Focal_loss_fixed_平衡采样1_10_(model_7auc_0.6659402744148506_f1_score_0.4210526315789474precision_0.5714285714285714recall_0.3333333333333333).h5',
        'LSTMAttention_预处理_Focal_loss_fixed_平衡采样1_10_(model_8auc_0.6454701372074253_f1_score_0.4117647058823529precision_0.7recall_0.2916666666666667).h5',
        'LSTMAttention_预处理_Focal_loss_fixed_平衡采样1_10_(model_9auc_0.6452278598530717_f1_score_0.38888888888888895precision_0.5833333333333334recall_0.2916666666666667).h5',
        'LSTMAttention_预处理_Focal_loss_fixed_平衡采样1_10_(model_10auc_0.6944412269525203_f1_score_0.42857142857142855precision_0.47368421052631576recall_0.391304347826087).h5']

        path3 =['LSTMAttention_预处理_Focal_loss_fixed_平衡采样1_1_(model_1auc_0.7079701372074253_f1_score_0.5405405405405406precision_0.7692307692307693recall_0.4166666666666667).h5',
        'LSTMAttention_预处理_Focal_loss_fixed_平衡采样1_1_(model_2auc_0.6041666666666666_f1_score_0.3448275862068966precision_1.0recall_0.20833333333333334).h5',
        'LSTMAttention_预处理_Focal_loss_fixed_平衡采样1_1_(model_3auc_0.6454701372074253_f1_score_0.4117647058823529precision_0.7recall_0.2916666666666667).h5',
        'LSTMAttention_预处理_Focal_loss_fixed_平衡采样1_1_(model_4auc_0.6457122679580307_f1_score_0.43750000000000006precision_0.875recall_0.2916666666666667).h5',
        'LSTMAttention_预处理_Focal_loss_fixed_平衡采样1_1_(model_5auc_0.6664245359160613_f1_score_0.47058823529411764precision_0.8recall_0.3333333333333333).h5',
        'LSTMAttention_预处理_Focal_loss_fixed_平衡采样1_1_(model_6auc_0.6873789346246973_f1_score_0.5294117647058825precision_0.9recall_0.375).h5',
        'LSTMAttention_预处理_Focal_loss_fixed_平衡采样1_1_(model_7auc_0.6454701372074253_f1_score_0.4117647058823529precision_0.7recall_0.2916666666666667).h5',
        'LSTMAttention_预处理_Focal_loss_fixed_平衡采样1_1_(model_8auc_0.666182405165456_f1_score_0.4444444444444444precision_0.6666666666666666recall_0.3333333333333333).h5',
        'LSTMAttention_预处理_Focal_loss_fixed_平衡采样1_1_(model_9auc_0.6660611931864051_f1_score_0.43243243243243246precision_0.6153846153846154recall_0.3333333333333333).h5',
        'LSTMAttention_预处理_Focal_loss_fixed_平衡采样1_1_(model_10auc_0.7380405825181378_f1_score_0.5116279069767442precision_0.55recall_0.4782608695652174).h5']

        modelpath1 = path1[cnt]
        modelpath2 = path2[cnt]
        modelpath3 = path3[cnt]

        model1 = CRNNAttentionModel([input_current,input_left,input_right],maxlen,max_token,embedding_matrix,embedding_dims=150)
        model1.load_weights("./model_openstack/"+modelpath1)

        model2 = AttBilstm(input_current,maxlen,max_token,embedding_matrix,embedding_dims=150)
        model2.load_weights("./model_openstack/"+modelpath2)

        model3 = AttBilstm(input_current,maxlen,max_token,embedding_matrix,embedding_dims=150)
        model3.load_weights("./model_openstack/"+modelpath3)

        val_predict = list(map(boolMap,model1.predict([x_test_current,x_test_left,x_test_right])*0.4+model2.predict(x_test_current)*0.3+model3.predict(x_test_current)*0.3))
        # val_predict2 = list(map(boolMap,model2.predict(x_test_current)))

        val_targ = y_train[test]

        auc = roc_auc_score(val_targ, val_predict,average=None)
        val_f1 = f1_score(val_targ, val_predict,average=None)
        val_recall = recall_score(val_targ, val_predict,average=None)
        val_precision = precision_score(val_targ, val_predict,average=None)        

        print("auc : ",auc)
        print("val_f1 : ",val_f1)
        print("val_recall : ",val_recall)
        print("val_precision : ",val_precision)

        f1_score_list.append(val_f1[1])
        auc_list.append(auc)
        precision_list.append(val_precision[1])
        recall_list.append(val_recall[1])
        cnt += 1

    print(numpy.mean(f1_score_list))
    print(numpy.mean(auc_list))
    print(numpy.mean(precision_list))
    print(numpy.mean(recall_list))

ensemble(X,Y,maxlen[0],max_token[0],embedding_matrix[0],embedding_dims)
