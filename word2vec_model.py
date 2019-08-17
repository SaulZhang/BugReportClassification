# -*- coding: utf-8 -*-
'''
2019-7-29 13:00 by @saulzhang
'''
import pandas as pd
import numpy as np
import gensim
import importlib
import sys
import pickle,pprint
import os
import csv


from nltk.stem.porter import PorterStemmer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

importlib.reload(sys)
embedding_dims = 150


def getData(path1,path2,w2vpath,word2idx_pkl):
    #读取csv文件
    csv_file_0 = csv.reader(open(path1))
    csv_file_1 = csv.reader(open(path2))
    word_list = []
    textSet = []
    labelSet = []
    
    porter_stemmer = PorterStemmer()

    for idx,(data,label) in enumerate(csv_file_0):
        # if idx > 30: break
        data = data.lower()
        data = data.translate(str.maketrans('', '',"★[！？。｡＂＃＄％＆＇（）＊＋，－：；＜＝＞＠［＼］＾｀｛｜｝～｟｠｢｣､、〃《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.!\"#$%&\'()*+,.;<=>?@[\\]^`{|}~]+" ))
        # print(data.lower()+"\n")
        labelSet.append(label)
        if(idx%1000==0):
            print("process line:",idx)
        # print(data+"\n")
        # print(data+"\n")
        li = []
        for token in data.split(" "):
            # token = porter_stemmer.stem(token)
            li.append(token)
            if token not in word_list:
                word_list.append(token)
        data = " ".join(str(i) for i in li)
        textSet.append(data)
        
    for idx,(data,label) in enumerate(csv_file_1):
        # if idx > 30: break
        data = data.lower()
        data = data.translate(str.maketrans('', '',"★[！？。｡＂＃＄％＆＇（）＊＋，－：；＜＝＞＠［＼］＾｀｛｜｝～｟｠｢｣､、〃《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.!\"#$%&\'()*+,.;<=>?@[\\]^`{|}~]+" ))
        # print(data.lower()+"\n")
        labelSet.append(label)
        if(idx%1000==0):
            print("process line:",idx)
        # print(data+"\n")
        # print(data+"\n")
        li = []
        for token in data.split(" "):
            # token = porter_stemmer.stem(token)
            li.append(token)
            if token not in word_list:
                word_list.append(token)
        data = " ".join(str(i) for i in li)
        textSet.append(data)

    print("len(word_list):",len(word_list))
    cnt = 0

    word_to_id = dict(zip(word_list, range(len(word_list))))
    word2idx_pkl_file = open(word2idx_pkl,'wb')
    pickle.dump(word_to_id,word2idx_pkl_file)

    embeddings_index = {}
    model = gensim.models.Word2Vec.load(w2vpath)


    #初始化一个0向量 统计未出现词个数
    null_word=np.zeros(embedding_dims)
    null_word_count=0

    for word in word_list:
        try:
            embeddings_index[word]=model[word]
        except:
            embeddings_index[word]=null_word
            null_word_count+=1
            # print("word=,",word)
    print('Found %s word vectors.' % len(embeddings_index))
    print('Found %s null word.' % null_word_count)
    print('Preparing embedding matrix.')
    max_token = len(word_to_id)
    embedding_matrix = np.zeros((max_token + 1, embedding_dims))
    for word, i in word_to_id.items():
        if i > max_token:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    print("embedding_matrix存储完毕")
    # print(embedding_matrix)

    import random
    cc = list(zip(textSet,labelSet))
    random.shuffle(cc)
    textSet,labelSet=zip(*cc)
    textSet = list(textSet)
    labelSet = list(labelSet)
    print(len(word_list))
    # print("word2idx:",word_to_id)
    dx=[]
    for idx,line in enumerate(textSet):
        # if idx>10:break
        ws=str(line).split(" ")
        li = []
        for w in ws :
            if w in word_list:#porter_stemmer.stem(w)
                li.append(word_to_id[w])#porter_stemmer.stem(w)
        dx.append(li)
        # dx.append([word_to_id[porter_stemmer.stem(w)] for w in ws if porter_stemmer.stem(w) in word_to_id])

    # print("dx:",dx)
    count = 0
    label_dict = {}
    dy = labelSet
    for idx,label in enumerate(dy):
        if label not in label_dict.keys():
            label_dict[label]= count
            count += 1

    for idx in range(len(dy)):
        dy[idx] = label_dict[dy[idx]]

    # print(dx)
    # print(dy)
    print('Average  sequence length: {}'.format(np.mean(list(map(len, dx)), dtype=int)))

    sortmaxlen=np.sort(list(map(len, dx)), axis=0)
    print(sortmaxlen)
    maxlen = sortmaxlen[int(len(dy)*0.8)]
    print("maxlen:",maxlen)
    num_classes = len(label_dict)
    print("num_classes:",num_classes)
    x_train, y_train, x_test, y_test = dx[0:int(len(dy)*0.5)],dy[0:int(len(dy)*0.5)],dx[int(len(dy)*0.5):],dy[int(len(dy)*0.5):]
    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')
    x_train = pad_sequences(x_train, maxlen=maxlen)
    x_test = pad_sequences(x_test, maxlen=maxlen)
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)

    x_train_dataset_pkl = '../data_openstack/x_train_dataset.pkl'
    y_train_dataset_pkl = '../data_openstack/y_train_dataset.pkl'
    x_test_dataset_pkl = '../data_openstack/x_test_dataset.pkl'
    y_test_dataset_pkl = '../data_openstack/y_test_dataset.pkl'

    x_train_dataset_pkl_file = open(x_train_dataset_pkl,'wb')
    pickle.dump(x_train,x_train_dataset_pkl_file)

    y_train_dataset_pkl_file = open(y_train_dataset_pkl,'wb')
    pickle.dump(y_train,y_train_dataset_pkl_file)

    x_test_dataset_pkl_file = open(x_test_dataset_pkl,'wb')
    pickle.dump(x_test,x_test_dataset_pkl_file)

    y_test_dataset_pkl_file = open(y_test_dataset_pkl,'wb')
    pickle.dump(y_test,y_test_dataset_pkl_file)

    other_info_pkl = '../data_openstack/other_info.pkl'
    other_info_file = open(other_info_pkl,'wb')
    pickle.dump([[maxlen],[max_token],[embedding_matrix]],other_info_file)

def main():
    
    w2vpath = '../word2vec_model/openstack/word2vec.model'
    path1 = '../dataset/input_sec/openstack_0.csv'
    path2 = '../dataset/input_sec/openstack_1.csv'
    word2idx_pkl = "../data_openstack/word2idx_dict.pkl"

    getData(path1,path2,w2vpath,word2idx_pkl)


if __name__ == '__main__':
    main()
