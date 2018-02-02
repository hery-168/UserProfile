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

from sklearn.feature_extraction.text import TfidfVectorizer

if __name__ == '__main__':
    pass