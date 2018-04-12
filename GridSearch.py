# -*- coding: utf-8 -*-
"""
Created on Mon Apr 09 15:54:22 2018

@author: yingliang.huang
"""

import pandas as pd
import numpy as np
import woe.feature_process as fp
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from datetime import datetime
from sklearn.svm import l1_min_c
from woe.eval import  compute_ks
import pickle
import time


def grid_search_lr_c_validation(X_train,y_train,validation_dataset_list,cs=[0.01],df_coef_path=False
                                ,pic_coefpath_title='Logistic Regression Path',pic_coefpath=False
                                ,pic_performance_title='Logistic Regression Performance',pic_performance=False):


    clf_l1_LR = LogisticRegression(C=0.1, penalty='l1', tol=0.01,class_weight='balanced')
    print("Computing regularization path ...")
    start = datetime.now()
    print(start)
    coefs_ = []
    ks = []
    ks_validation1 = []
    counter = 0
    for c in cs:
        print('time: ',time.asctime(time.localtime(time.time())),'counter: ',counter, ' c: ',c)
        clf_l1_LR.set_params(C=c)
        clf_l1_LR.fit(X_train, y_train)
        coefs_.append(clf_l1_LR.coef_.ravel().copy())
        proba = clf_l1_LR.predict_proba(X_train)[:,1]
        validation_proba1 = clf_l1_LR.predict_proba(validation_dataset_list[0][X_train.columns])[:,1]
        a1 = pd.DataFrame(np.array([proba,y_train]).T,columns=['proba','target'])
        a2 = pd.DataFrame(np.array([validation_proba1,validation_dataset_list[0]['target']]).T,columns=['proba','target'])
        ks.append(compute_ks(a1['target'],a1['proba']))
        ks_validation1.append(compute_ks(a2['target'],a2['proba']))
        print('ks:\t',ks[-1],'ks_validation1:\t',ks_validation1[-1])
        counter += 1
    end = datetime.now()
    print(end)
    print("This took ", end - start)
    coef_cv_df = pd.DataFrame(coefs_,columns=X_train.columns)
    coef_cv_df['ks'] = ks
    coef_cv_df['ks_validation1'] = ks_validation1
    coef_cv_df['c'] = cs

    if df_coef_path:
        file_name = df_coef_path if isinstance(df_coef_path, str) else None
        coef_cv_df.to_csv(file_name)

    coefs_ = np.array(coefs_)
    fig1 = plt.figure('fig1')
    plt.plot(np.log10(cs), coefs_)
    ymin, ymax = plt.ylim()
    plt.xlabel('log(C)')
    plt.ylabel('Coefficients')
    plt.title(pic_coefpath_title)
    plt.axis('tight')

    if pic_coefpath:
        file_name = pic_coefpath if isinstance(pic_coefpath, str) else None
        plt.savefig(file_name)
        plt.close()
    else:
        pass

    fig2 = plt.figure('fig2')
    plt.plot(np.log10(cs), ks)
    plt.xlabel('log(C)')
    plt.ylabel('ks score')
    plt.title(pic_performance_title)
    plt.axis('tight')

    if pic_performance:
        file_name = pic_performance if isinstance(pic_performance, str) else None
        plt.savefig(file_name)
        plt.close()
    else:
        pass

    flag = coefs_<0
    if np.array(ks)[flag.sum(axis=1) == 0].__len__()>0:
        idx = np.array(ks)[flag.sum(axis=1) == 0].argmax()
    else:
        idx = np.array(ks).argmax()
    return (cs[idx],ks[idx])





def grid_search_lr_c_main(params):
    print('run into grid_search_lr_c_main:')
    dataset_path = params['dataset_path']
    validation_path = params['validation_path']
    config_path = params['config_path']
    df_coef_path = params['df_coef_path']
    pic_coefpath = params['pic_coefpath']
    pic_performance = params['pic_performance']
    pic_coefpath_title = params['pic_coefpath_title']
    pic_performance_title = params['pic_performance_title']
    dataset_train = pd.read_csv(dataset_path)
    cfg = pd.read_csv(config_path)
    candidate_var_list = cfg[cfg['is_modelfeature'] == 1]['var_name']
    b = [var for var in dataset_train.columns if sum(dataset_train[var].isnull()) == 0]
    candidate_var_list = list(set(candidate_var_list).intersection(set(b)))
    var_list_specfied = params['var_list_specfied']
    if var_list_specfied.__len__()>0:
        candidate_var_list = list(set(candidate_var_list).intersection(set(var_list_specfied)))

    print('candidate_var_list length:\n',candidate_var_list.__len__())
    print('candidate_var_list:\n',candidate_var_list)
    print('change dtypes:float64 to float32')

    for var in candidate_var_list:
        dataset_train[var] = dataset_train[var].astype(np.float32)
    X_train = dataset_train[dataset_train.target >=0][candidate_var_list]
    y_train = dataset_train[dataset_train.target >=0]['target']

    validation_cols_keep = [var for var in candidate_var_list]
    validation_cols_keep.append('target')
    validation_dataset_list = []
    validation_dataset = pd.read_csv(validation_path)

    for var in candidate_var_list:
        validation_dataset.loc[validation_dataset[var].isnull(), (var)] = 0
    validation_dataset_list.append(validation_dataset[validation_cols_keep])
    cs = params['cs']
    print('cs',cs)
    c,ks = grid_search_lr_c_validation(X_train,y_train,validation_dataset_list,cs,df_coef_path,pic_coefpath_title,pic_coefpath
                                       ,pic_performance_title,pic_performance)
    print('pic_coefpath:\n',pic_coefpath)
    print('pic_performance:\n',pic_performance)
    print('ks performance on the c:')
    print(c,ks)
    return (c,ks)



def fit_single_lr(dataset_path,config_path,var_list_specfied,out_model_path,c=0.01):
    dataset_train = pd.read_csv(dataset_path)
    cfg = pd.read_csv(config_path)
    candidate_var_list = cfg[cfg['is_modelfeature'] == 1]['var_name']
    b = [var for var in dataset_train.columns if sum(dataset_train[var].isnull()) == 0]
    candidate_var_list = list(set(candidate_var_list).intersection(set(b)))
    if var_list_specfied.__len__()>0:
        candidate_var_list = list(set(candidate_var_list).intersection(set(var_list_specfied)))
    print('candidate_var_list length:\n',candidate_var_list.__len__())
    print('candidate_var_list:\n',candidate_var_list)
    print('change dtypes:float64 to float32')

    for var in candidate_var_list:
        dataset_train[var] = dataset_train[var].astype(np.float32)

    X_train = dataset_train[dataset_train.target >=0][candidate_var_list]
    y_train = dataset_train[dataset_train.target >=0]['target']
    print('c:',c)

    clf_lr_a = LogisticRegression(C=c, penalty='l1', tol=0.01,class_weight='balanced')
    clf_lr_a.fit(X_train, y_train)
    coefs = clf_lr_a.coef_.ravel().copy()
    proba = clf_lr_a.predict_proba(X_train)[:,1]
    a = pd.DataFrame(np.array([proba,y_train]).T,columns=['proba','target'])
    ks = compute_ks(a['target'],a['proba'])

    model = {}
    model['clf'] = clf_lr_a
    model['features_list'] = candidate_var_list
    model['coefs'] = coefs
    model['ks'] = ks

    output = open(out_model_path, 'wb')
    pickle.dump(model,output)
    output.close()

    return model