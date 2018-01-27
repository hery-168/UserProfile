#-*- coding:utf-8 _*-

"""
@version: 1.0
@author: CharlesXu
@license: Q_S_Y_Q 
@file: Create_features_A.py
@time: 2018/1/23 10:15
@desc: 根据数据构建特征
       构建低敏感度特征：在工单中只有一条通话记录
"""

import pandas as pd
import numpy as np
import csv
import re
import jieba
import codecs      # 专门做编码转换
import pickle

from numpy import log
from sklearn.preprocessing import MinMaxScaler  # 将属性缩放到一个指定范围。这种方法的目的包括:
                                                # 1,对于方差非常小的属性可以增强其稳定性
                                                # 2,维持稀疏矩阵中为0的条目
                                                # 在构造类对象的时候也可以直接指定最大最小值的范围: feature_range=(min, max)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder # LabelEncoder 对不连续的文本或者数字进行编号
                                                              # OneHotEncoder 用于将表示分类的数据进行扩维
from sklearn.preprocessing import scale   # 直接将数据进行标准化  将数据按期属性（按列进行）减去其均值，并处以其方差。
from sklearn.preprocessing import StandardScaler # 使用该类的好处就是可以保存训练集中的参数(均值，方差)直接使用期对象转换测试集数据
from sklearn.preprocessing import Normalizer     # 对训练集和测试集的拟合和转换


'''
核心数据介绍:
   表名称                                 数据概况         
01 95598工单信息                  包含全部训练集和测试集，核心数据
02 客户通话信息记录                包含训练集用户656641个，测试集用户369560个
09 应收电费信息表                  包含训练集用户555748个，测试集用户201702个
'''


data_path_train = 'E:\py_workspace\\UserProfile\data\\train/'  # 训练数据文件夹
data_path_test = 'E:\py_workspace\\UserProfile\data\\test/'    # 测试数据文件夹
# train
file_jobInfo_train = '01_arc_s_95598_wkst_train.tsv'   # 01是95598工单信息，包含全部训练集和测试集，核心数据
# test
file_jobInfo_test = '01_arc_s_95598_wkst_test.tsv'
# 通话信息记录
file_comm = '02_s_comm_rec.tsv'
file_flow_train = '09_arc_a_rcvbl_flow.tsv'
file_flow_test = '09_arc_a_rcvbl_flow_test.tsv'
# 训练集正例
file_label = 'train_label.csv'
# 测试集
file_test = 'test_to_predict.csv'

print('Processing Data...')
# ------加载数据------
with codecs.open(data_path_train + file_jobInfo_train, 'r', encoding='utf-8') as fin, \
     codecs.open(data_path_train + 'processed_' + file_jobInfo_train, 'w', encoding='utf-8') as fout:
     for index, line in enumerate(fin):
         items = line.strip().split('\t')
         for i, item in enumerate(items):
             item = item.strip()
             if i < 12:
                 fout.write(item + '\t')
             else:
                 fout.write(item + '\n')
     print('() lines in train_95598'.format(index))

with codecs.open(data_path_test + file_jobInfo_test, 'r', encoding='utf-8') as fin,\
     codecs.open(data_path_test + 'processed_' + file_jobInfo_test, 'w', encoding='utf-8') as fout:
    for index, line in enumerate(fin):
        items = line.strip().split('\t')
        for i, item in enumerate(items):
            item = item.strip()
            if i < 12:
                fout.write(item + '\t')
            else:
                fout.write(item + '\n')
    print('() lines in test_95598'.format(index))

# 处理训练数据
train_info = pd.read_csv(data_path_train + 'processed_' + file_jobInfo_train, sep='\t', encoding='utf-8', quoting=csv.QUOTE_NONE)
# 过滤 CUST_NO为空的用户
train_info = train_info.loc[~train_info.CUST_NO.isnull()]  # loc  通过行标签索引行数据
                                                           # iloc 通过行号索引行数据
                                                           # ix   通过行标签或者行号索引行数据(loc 和 iloc的混合)
train_info['CUST_NO'] = train_info.CUST_NO.astype(np.int64)  # astype 实现变量类型转换

# 构建用户索引
train = train_info.CUST_NO.value_counts().to_frame().reset_index()
train.columns = ['CUST_NO', 'counts_of_jobinfo']
temp = pd.read_csv(data_path_train + file_label, header=None)
temp.columns = ['CUST_NO']
train['label'] = 0
train.loc[train.CUST_NO.isin(temp.CUST_NO), 'label'] = 1
train = train[['CUST_NO', 'label', 'counts_of_jobinfo']]

test_info = pd.read_csv(data_path_test + 'processed_' + file_jobInfo_test, sep='\t', encoding='utf-8', quoting=csv.QUOTE_NONE)
test = test_info.CUST_NO.value_counts().to_frame().reset_index()
test.columns = ['CUST_NO', 'counts_of_jobinfo']
test['label'] = -1
test = test[['CUST_NO', 'label', 'counts_of_jobinfo']]

df = train.append(test).copy()
del temp, train, test

############
# 只保留一条工单信息的低敏感度用户
############
df = df.loc[df.counts_of_jobinfo == 1].copy()
df.reset_index(drop=True, inplace=True)
train = df.loc[df.label != 1]
test = df.loc[df.label == -1]
print('原始数据中的低敏感度用户分布情况如下：')
print('训练集:', train.shape[0])
print('正样本:', train.loc[train.label == 1].shape[0])
print('负样本:', train.loc[train.label == 0].shape[0])
print('---------------')
print('测试集:', test.shape[0])
df.drop(['counts_of_jobinfo'], axis=1, inplace=True)


#   --------- 构建特征 -------
# 读取表2   客户通话信息记录
# 没有表2信息的用户全是非敏感用户
print('Creating Features...')
# 合并工单
jobinfo = train_info.append(test_info).copy()
jobinfo = jobinfo.loc[jobinfo.CUST_NO.isin(df.CUST_NO)].copy()
jobinfo.reset_index(drop=True, inplace=True)
 #python的merge函数具体参见博客: http://blog.csdn.net/ly_ysys629/article/details/73849543
jobinfo = jobinfo.merge(df[['CUST_NO', 'label']], on='CUST_NO', how='left')

#  ######
print('处理表2...')
comm = pd.read_csv(data_path_train + file_comm, sep='\t')
comm.drop_duplicates(inplace=True)   # 去重 inplace=True 是直接对原dataFrame进行操作。
                                     #     inplace=False 不改变原来的dataFrame，而将结果生成在一个新的dataFrame中。
comm = comm.loc[comm.APP_NO.isin(jobinfo.ID)]
comm = comm.rename(columns={'APP_NO':'ID'})
comm = comm.merge(jobinfo[['ID', 'CUST_NO']], on='ID', how='left')
comm['REQ_BEGIN_DATE'] = comm.REQ_BEGIN_DATE.apply(lambda x : pd.to_datetime(x))
comm['REQ_FINISH_DATE'] = comm.REQ_FINISH_DATE.apply(lambda x : pd.to_datetime(x))

# 过滤
comm = comm.loc[~(comm.REQ_BEGIN_DATE > comm.REQ_FINISH_DATE)]
df = df.loc[df.CUST_NO.isin(comm.CUST_NO)].copy()
comm['holding_time'] = comm['REQ_FINISH_DATE'] - comm['REQ_BEGIN_DATE']
comm['holding_time_seconds'] = comm.holding_time.apply(lambda x:x.seconds)









