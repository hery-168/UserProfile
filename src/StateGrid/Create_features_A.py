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
import os
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
df = df.merge(comm[['CUST_NO', 'holding_time_seconds']], how='left', on='CUST_NO')

# 通话时长归一化，秒
# fit_transform()和transform比较: 只传一个参数是无监督学习算法，比如降维，特征提取，标准化
# transform函数可以替换fit_transform(). fit_transform()不能替换transform()
# fit_transform()先拟合数据，再标准化
# transform()通过找中心和缩放等实现标准化
df['holding_time_seconds'] = MinMaxScaler().fit_transform(df['holding_time_seconds']) # 要带一个参数
del comm

print('处理表1...')
jobinfo = jobinfo.loc[jobinfo.CUST_NO.isin(df.CUST_NO)].copy()
jobinfo.reset_index(drop=True, inplace=True)

# 排序
df['rank_CUST_NO'] = df.CUST_NO.rank(method='max')
df['rank_CUST_NO'] = MinMaxScaler.fit_transform()

# 转换成独热向量
df = df.merge(jobinfo[['CUST_NO', 'BUSI_TYPE_CODE']], on='CUST_NO', how='left')
temp = pd.get_dummies(df.URBAN_RURAL_FLAG, prefix='onthot_URBAN_RURAL_FLAG', dummy_na=True) # 返回一个矩阵
df = pd.concat([df, temp], axis=1)
df.drop(['URBAN_RURAL_FLAG'], axis=1, inplce=True)
del temp

# 处理org_no
# ORG_NO编码
# 12个一级编码，5位数字：33401，33402，.... ,33410， 33411， 33420
# 75个二级编码，7位数字：前缀（33401，33402，.... ,33410， 33411）
# 96个三级编码，9位数字：前缀（33401，33402，.... ,33410， 33411）
# 1个四级编码，11位数字：33406400142
# add
df = df.merge(jobinfo[['CUST_NO', 'ORG_NO']], on='CUST_NO', how='left')
df['len_of_ORG_NO'] =df.ORG_NO.apply(lambda  x: len(str(x)))
df.fillna(-1, inplce=True)
# ratio
train = df[df.label != -1]
ratio = {}
for i in train.ORG_NO.unique():
    ratio[i] = len(train.loc[(train.ORG_NO == i) & (train.label == 1)]) / len(train.loc[train.ORG_NO == i])
df['ratio_ORG_NO'] = df.ORG_NO.map(ratio)
# one-hot
temp = pd.get_dummies(df.len_of_ORG_NO, prefix='onehot_len_of_ORG_NO')
df = pd.concat([df, temp], axis=1)
# drop
df.drop(['ORG_NO', 'len_of_ORG_NO'], axis=1, inplace=True)

# 处理时间
#  add
df = df.merge(jobinfo[['CUST_NO', 'HANDLE_TIME']], on='CUST_NO', how='left')
df['date'] = df['HANDLE_TIME'].apply(lambda x : pd.to_datetime(x.split()[0]))
df['time'] = df['HANDLE_TIME'].apply(lambda x : x.split()[1])
# day
df['day'] = df.date.apply(lambda x : x.day)
# 上旬，中旬，下旬
df['is_in_first_tendys'] = 0
df.loc[df.day.isin(range(1,11)), 'is_in_first_tendays'] = 1
df['is_in_middle_tendays'] = 0
df.loc[df.day.isin(range(11,21)), 'is_in_middle_tendays'] = 1
df['is_in_last_tendys']
df.loc[df.day.isin(range(21,32)), 'is_in_last_tendys'] = 1
# hour
# label encoder
df['hour'] = df.time.apply(lambda x:int(x.split(':')[0]))
# drop
df.drop(['HANDLE_TIME', 'date', 'time'], axis=1, inplce=True)
# Elec_type字段
df = df.merge(jobinfo[['CUST_NO', 'ELEC_TYPE']], on='CUST_NO', how='left')
df.fillna(0, inplace=True)
df['head_of_ELEC_TYPE'] = df.ELEC_TYPE.apply(lambda x:str(x)[0])
# 判断是否是空值
df['is_ELEC_TYPE_NaN'] = 0
df.loc[df.ELEC_TYPE == 0, 'is_ELEC_TYPE_NaN'] = 1
# label encoder
df['label_encoder_ELEC_TYPE'] = LabelEncoder().fit_transform(df['ELEC_TYPE'])
# ratio
train = df[df.label != -1]
ratio = {}
for i in train.ELEC_TYPE.unique():
    ratio[i] = len(train.loc[(train.ELEC_TYPE == i) & (train.label == 1)]) / len(train.loc[train.ELEC_TYPE == i])
df['ratio_ELEC_TYPE'] = df.ELEC_TYPE.map(ratio)
df.fillna(0, inplce=True)

# 3,用电类别第一位数字 one-hot
temp = pd.get_dummies(df.head_of_ELEC_TYPE, prefix='onehot_head_of_ELEC_TYPE')
df = pd.concat([df. time], axis=1)
df.drop(['ELEC_TYPE', 'hed_of_ELEC_TYPE'], axis=1, inplace=True)

# CITY_ORG_NO 字段
df.merge(jobinfo[['CUST_NO', 'CITY_ORG_NO']], on='CUST_NO', how='left')
train = df[df.label != -1]
ratio ={}
for i in train.CITY_ORG_NO.unique():
    ratio[i] = len(train.loc[(train.CITY_ORG_NO == i) & (train.label == 1)]) / len(train.loc[train.CITY_ORG_NO == 1])
df['ratio_CITY_ORG_NO'] = df.CITY_ORG_NO.map(ratio)
# one-hot
temp = pd.get_dummies(df.CITY_ORG_NO, prefix='onehot_CITY_ORG_NO')
df = pd.concat([df, temp], axis=1)
# drop
df.drop(['CITY_ORG_NO'], axis=1, inplace=True)

# 处理 应收电费信息表
train_flow = pd.read_csv(data_path_train + file_flow_train, sep='\t')
test_flow = pd.read_csv(data_path_test + file_flow_test, sep='\t')
flow = train_flow.append(test_info).copy()
flow.rename(columns={'CONS_NO': 'CUST_NO'}, inplace=True)
flow.drop_duplicates(inplace=True)
flow = flow.loc[flow.CUST_NO.isin(df.CUST_NO)].copy()

flow['T-PQ'] = flow.T_PQ.apply(lambda x: -x if x < 0 else x)
flow['RCVBL_AMT'] = flow.RCVBL_AMT.apply(lambda x: -x if x < 0 else x)
flow['RCVED_AMT'] = flow.RCVED_AMT.apply(lambda x: -x if x < 0 else x)
flow['OWE_AMT'] = flow.OWE_AMT.apply(lambda x: -x if x < 0 else x)

# 判断是否有表9
df['has_nine'] = 0
df.loc[df.CUST_NO.isin(flow.CUST_NO), 'has_nine'] = 1
df['counts_of_09flow'] = df.CUST_NO.map(flow.groupby('CUST_NO').size())

# 应收金额
df['sum_yingshoujine'] = log(df.CUST_NO.map(flow.groupby('CUST_NO').RCVBL_AMT.sum()) + 1)    # 求和
df['mean_yingshoujine'] = log(df.CUST_NO.map(flow.groupby('CUST_NO').RCVBL_AMT.mean()) + 1)  # 求均值
df['max_yingshoujine'] = log(df.CUST_NO.map(flow.groupby('CUST_NO').RCVBL_AMT.max()) + 1)
df['min_yingshoujine'] = log(df.CUST_NO.map(flow.groupby('CUST_NO').RCVBL_AMT.min()) + 1)
df['std_yingshoujine'] = log(df.CUST_NO.map(flow.groupby('CUST_NO').RCVBL_AMT.std()) + 1)    # 标准差

# 实收金额
df['sum_shishoujine'] = log(df.CUST_NO.map(flow.groupby('CUST_NO').RCVED_AMT.sum()) + 1)

# 应收 - 实收 = 欠费
df['qianfei'] = df['sum_yingshoujine'] = df['sum_shishoujine']

# 总电量
df['sum_T_PQ'] = log(df.CUST_NO.map(flow.groupby('CUST_NO').T_PQ.sum()) + 1)
df['mean_T_PQ'] = log(df.CUST_NO.map(flow.groupby('CUST_NO').T_PQ.mean()) + 1)
df['max_T_PQ'] = log(df.CUST_NO.map(flow.groupby('CUST_NO').T_PQ.max()) + 1)
df['min_T_PQ'] = log(df.CUST_NO.map(flow.groupby('CUST_NO').T_PQ.min()) + 1)
df['std_T_PQ'] = log(df.CUST_NO.map(flow.groupby('CUST_NO').T_PQ.std()) + 1)

# 电费金额
df['sum_OWE_AMT'] = log(df.CUST_NO.map(flow.groupby('CUST_NO').OWE_AMT.sum()) + 1)
df['mean_OWE_AMT'] = log(df.CUST_NO.map(flow.groupby('CUST_NO').OWE_AMT.mean()) + 1)
df['max_OWE_AMT'] = log(df.CUST_NO.map(flow.groupby('CUST_NO').OWE_AMT.max()) + 1)
df['min_OWE_AMT'] = log(df.CUST_NO.map(flow.groupby('CUST_NO').OWE_AMT.min()) + 1)
df['std_OWE_AMT'] = log(df.CUST_NO.map(flow.groupby('CUST_NO').OWE_AMT.std()) + 1)

# 电费金额和应收金额差多少
df['dianfei_jian_yingshoujine'] = df['sum_OWE_AMT'] - df['sum_yingshoujine']

# 应收违约金
df['sum_RCVBL_PENALTY'] = log(df.CUST_NO.map(flow.groupby('CUST_NO').RCVBL_PENALTY.sum()) + 1)
df['mean_RCVBL_PENALTY'] = log(df.CUST_NO.map(flow.groupby('CUST_NO').RCVBL_PENALTY.mean()) + 1)
df['max_RCVBL_PENALTY'] = log(df.CUST_NO.map(flow.groupby('CUST_NO').RCVBL_PENALTY.max()) + 1)
df['min_RCVBL_PENALTY'] = log(df.CUST_NO.map(flow.groupby('CUST_NO').RCVBL_PENALTY.min()) + 1)
df['std_RCVBL_PENALTY'] = log(df.CUST_NO.map(flow.groupby('CUST_NO').RCVBL_PENALTY.std()) + 1)

# 实收违约金
df['sum_RCVED_PENALTY'] = log(df.CUST_NO.map(flow.groupby('CUST_NO').RCVED_PENALTY.sum()) + 1)
df['mean_RCVED_PENALTY'] = log(df.CUST_NO.map(flow.groupby('CUST_NO').RCVED_PENALTY.mean()) + 1)
df['max_RCVED_PENALTY'] = log(df.CUST_NO.map(flow.groupby('CUST_NO').RCVED_PENALTY.max()) + 1)
df['min_RCVED_PENALTY'] = log(df.CUST_NO.map(flow.groupby('CUST_NO').RCVED_PENALTY.min()) + 1)
df['std_RCVED_PENALTY'] = log(df.CUST_NO.map(flow.groupby('CUST_NO').RCVED_PENALTY.std()) + 1)

df['chaduoshao_weiyuejin'] = df['sum_RCVBL_PENALTY'] - df['sum_RCVED_PENALTY']

# 每个用户有几个月的记录
df['nunique_RCVBL_YM'] = df.CUST_NO.map(flow.groupby('CUST_NO').RCVED_PENALTY.nunique())

# 平均每个月有几条
df['mean_RCVBL_YM'] = df['counts_of_09flow'] / df['nunique_RCVBL_YM']
del train_flow, test_flow, flow

if not os.path.isdir('../myFeatures'):
    os.makedirs('../myFeatures')

print('统计特征搞定!')
pickle.dump(df, open('../myFeatures/statistical_features_1.pkl', 'wb'))
print('开始处理表1中的文本特征...')

mywords = ['户号', '分时', '抄表', '抄表示数', '工单', '单号', '工单号', '空气开关', '脉冲灯', '计量表', '来电', '报修']
for word in mywords:
    jieba.add_word(word)  # 可在程序中动态修改词典

stops = set()
stopwords = 'E:\py_workspace\\UserProfile\stopwords.txt'
with open(stopwords, encoding='utf-8') as f:
    for word in f:
        word = word.strip()
        stops.add(word)

def fenci(line):
    res = []
    words = jieba.cut(line)
    for word in words:
        if word not in stops:
            res.append(word)
    return ' '.join(res)
print('分词ing...')
jobinfo['contents'] = jobinfo.ACCEPT_CONTENT.apply(lambda x:fenci(x))

def hash_numbers(x):
    shouji_pattern = re.compile('\s1\d{10}\s|\s1\d{10}\Z')
    if shouji_pattern.findall(x):
        x = re.sub(shouji_pattern, '手机number', x)

    huhao_pattern = re.compile('\s\d{10}\s|\s\d{10}\Z')
    if huhao_pattern.findall(x):
        x = re.sub(huhao_pattern, ' 户号number ', x)

    tuiding_pattern = re.compile('\s\d{11}\s|\s\d{11}\Z')
    if tuiding_pattern.findall(x):
        x = re.sub(tuiding_pattern, ' 退订number ', x)

    gongdan_pattern = re.compile('\s201\d{13}\s|\s201\d{13}\Z')
    if gongdan_pattern.findall(x):
        x = re.sub(gongdan_pattern, ' 工单number ', x)

    tingdian_pattern = re.compile('\s\d{12}\s|\s\d{12}\Z')
    if tingdian_pattern.findall(x):
        x = re.sub(tingdian_pattern, ' 停电number ', x)

    return x.strip()
jobinfo['content'] = jobinfo['contents'].apply(lambda x:hash_numbers(x))

jobinfo['len_of_contents'] = jobinfo.content.apply(lambda x:len(x.split()))
jobinfo['counts_of_words'] = jobinfo.content.apply(lambda x:len(set(x.split())))
text = df[['CUST_NO']].copy()
text = text.merge(jobinfo[['CUST_NO', 'len_of_contents', 'counts_of_words', 'content']], on='CUST_NO', how='left')
text = text.rename(columns={'content': 'contents'})

pickle.dump(text, open('../myfeatures/text_features_1.pkl', 'wb'))
print('done!')




















