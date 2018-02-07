#-*- coding:utf-8 _*-

"""
@version: 
@author: CharlesXu
@license: Q_S_Y_Q 
@file: Select_features_A.py
@time: 2018/2/2 16:48
@desc: 采用xgboost选择低敏感度用户的文本特征
"""

import pickle
import pandas as pd
import xgboost as xgb

from sklearn.feature_extraction.text import TfidfVectorizer  # 可以把文档转换成TF-IDF特征的矩阵， 将原始文档的集合转换为TF-IDF功能的矩阵
from scipy.sparse import csc_matrix    # 构造稀疏矩阵
from scipy.sparse import hstack

print('Select Features....')

df = pickle.load(open(' ', 'rb'))
text = pickle.load(open(' ', 'rb'))
df = df.merge(text, on='CUST_NO', how='left')

train = df.loc[df.label != -1]
test = df.loc[df.label == -1]

x_data = train.copy()
x_val = test.copy()

# 打乱训练集
x_data = x_data.sample(frac=1, random_state=1).reset_index(drop=True)
print('--------------')

delete_columns = ['CUST_NO', 'label', 'contents']

# 构造稀疏矩阵
X_train_1 = csc_matrix(x_data.drop(delete_columns, axis=1).as_matrix())       # 压缩稀疏列格式   # 注意csc_matrix() 和 csr_matrix()的区别
X_val_1 = csc_matrix(x_val.drop(delete_columns, axis=1).as_matrix())

y_train = x_data.label.values
y_val = x_val.label.values
featureNames = list(x_data.drop(delete_columns, axis=1).columns)

print('creating tfidf...')
tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=3, use_idf=False, smooth_idf=False, sublinear_tf=True)  # 添加参数注释
tfidf.fit(x_data.contents)

word_names = tfidf.get_feature_names()

X_train_2 = tfidf.transform(x_data.contents)
X_val_2 = tfidf.transform(x_val.contents)
print('文本特征: {}维'.format(len(word_names)))

statistic_feature = featureNames.copy()
print('其他特征: {}维'.format(len(statistic_feature)))
featureNames.extend(word_names)   #

X_train = hstack(((X_train_1), (X_train_2))).tocsc()

print('特征数量', X_train.shape[1])
print('----------------------')

############
##Xgboost##
print('采用xgboost筛选文本特征')
print('Training...')
dtrain = xgb.DMatrix(X_train, y_train, feature_names=featureNames)
dval = xgb.DMatrix(x_val, feature_names=featureNames)

params = {
    "objective": "binary:logistic",
    "booster": "gbtree",
    "eval_metric": "error",
    "eta": 0.1,
    'max_depth':12,
    'subsample':0.8,
    'min_child_weight':3,
    'colsample_bytree':1,
    'gamma':0.2,
    "lambda":300,
    "silent":1,
    'seed':1,
}
watchlist = [(dtrain, 'train')]
model = xgb.train(params, dtrain, 2500, evals=watchlist, early_stopping_rounds=100, verbose_eval=100)  #

print('训练完毕')
temp = pd.DataFrame.from_dict(model.get_fscore(), orient='index').reset_index()
temp.columns = ['feature', 'score']
temp.sort_values(['score'], axis=0, ascending=False, inplace=True)
temp.reset_index(drop=True, inplace=True)

print('留下的文本特征数量:', len(temp.loc[~temp.feature.isin(statistic_feature)]))

selected_words = list(temp.loc[~temp.feature.isin(statistic_feature)].feature.values)
pickle.dump(selected_words, open(' ', 'rb'))

