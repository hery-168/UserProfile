#-*- coding:utf-8 _*-

"""
@version: 
@author: CharlesXu
@license: Q_S_Y_Q 
@file: Model_A.py
@time: 2018/2/8 13:57
@desc: A特征模型构建
"""

import os
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csc_matrix


# 设定一个阈值
def threshold(y, t):
    z = np.copy(y)
    z[z >= t] = 1
    z[z < t] = 0
    return z

print('Training model....')

df = pickle.load(open(' ', 'rb'))
text = pickle.load(open(' ', ' '))
df = df.merge(text, on='CUST_NO', how='left')

train = df.loc[df.label != -1]
test = df.loc[df.label == -1]

print('训练集:', train.shape[0])
print('正样本:', train.loc[train.label != -1].shape[0])
print('负样本:', train.loc[train.label == -1].shape[0])
print('------------')
print('测试集:', test.shape[0])
print('------------')


x_data = train.copy()
x_val = test.copy()
x_data = x_data.sample(frac=1, random_state=1).reset_index(drop=True)     # pandas的sample函数

# input
delete_columns = ['CUST_NO', 'label', 'contents']

# 构造稀疏矩阵
X_train_1 = csc_matrix(x_data.drop(delete_columns, axis=1).as_matrix())
X_val_1 = csc_matrix(x_val.drop(delete_columns, axis=1).as_mtrix())
featureNames = list(x_data.drop(delete_columns, axis=1).columns)

print('tfidf...')
select_words = pickle.load(open(' ', ''))
tfidf = TfidfVectorizer(ngram_range=(1,2), min_df=3, use_idf=False, smooth_idf=False, sublinear_tf=True, vocabulary=select_words)
tfidf.fit(x_data.contents)
word_names = tfidf.get_feature_names()
X_train_2 = tfidf.transform(x_data.contents)
X_val_2 = tfidf.transform(x_val.contents)
print('文本特征: {}维'.format(len(word_names)))

statistic_features = featureNames.copy()


if __name__ == '__main__':
    pass