import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math
import random
from posixpath import split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif, f_classif, chi2
# load dataset

df = pd.read_csv("Data.csv", header = 0, delimiter = ";")

def drop_min_rows(df, *values):
    for value in values:
        df = df.drop(df[df['DP'] == value].index, axis=0)
    return df


def balance_classes(df):
    df_balanced = pd.DataFrame()
    min_class_count = df['DP'].value_counts().min()
    for label in df['DP'].unique():
        df_label = df[df['DP'] == label]
        if len(df_label) > min_class_count:
            df_random_sample = df_label.sample(min_class_count)
            df_balanced = pd.concat([df_balanced, df_random_sample])
        else:
            df_balanced = pd.concat([df_balanced, df_label])
    return df_balanced

def select_best_features(df_check):
    kbest = SelectKBest(k=10, score_func=mutual_info_classif)
    kbest.fit(df_check.drop('DP', axis=1), df_check['DP'])
    rec = kbest.get_support()
    list_best = []
    for i, r in enumerate(rec):
        if r == True:
            list_best.append(df_check.columns[i])
            print(df_check.columns[i], kbest.scores_[i])
    if 'DP' in list_best:
        list_best.remove('DP')
    df_best = df_check[list_best]
    return df_best


def data_preparation(df):
    df = df.drop_duplicates()
    constant_features = [column for column in df.columns if df[column].nunique() == 1]
    df.drop(columns=constant_features, inplace=True)
    df = pd.concat([df_corr, df_target], axis=1)
    new_df = drop_min_rows(df, -60, -140)
    new_df = balance_classes(new_df)
    check_class_disbalance(new_df, 'DP')
    df_check = new_df.copy()
    df_best = select_best_features(df_check)
    if df.Disbalance.empty == None:
        df_best =  pd.concat([df_check.Disbalance, df_best], axis=1)
    return df_best

df_ready = data_preparation(df)
print(df_ready.shape())
    
