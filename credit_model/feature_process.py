# -*- coding: utf-8 -*-
"""
Created on Sun Apr 08 15:01:08 2018

@author: yingliang.huang
"""

import numpy as np
import credit_model.config as config
import credit_model.eval as eval
import copy
import pickle
import time

class node:
    '''Tree Node Class
    '''
    def __init__(self,var_name=None,iv=0,split_point=None,right=None,left=None):
        self.var_name = var_name  # The column index value of the attributes that are used to split data sets
        self.iv = iv  # The info value of the node
        self.split_point = split_point  # Store split points list
        self.right = right  # Right sub tree
        self.left = left  # Left sub tree

class InfoValue(object):
    '''
    InfoValue Class
    '''
    def __init__(self):
        self.var_name = []
        self.split_list = []
        self.iv = 0
        self.woe_list = []
        self.iv_list = []
        self.is_discrete = 0
        self.sub_total_sample_num = []
        self.positive_sample_num = []
        self.negative_sample_num = []
        self.sub_total_num_percentage = []
        self.positive_rate_in_sub_total = []
        self.negative_rate_in_sub_total = []

    def init(self,civ):
        self.var_name = civ.var_name
        self.split_list = civ.split_list
        self.iv = civ.iv
        self.woe_list = civ.woe_list
        self.iv_list = civ.iv_list
        self.is_discrete = civ.is_discrete
        self.sub_total_sample_num = civ.sub_total_sample_num
        self.positive_sample_num = civ.positive_sample_num
        self.negative_sample_num = civ.negative_sample_num
        self.sub_total_num_percentage = civ.sub_total_num_percentage
        self.positive_rate_in_sub_total = civ.positive_rate_in_sub_total
        self.negative_rate_in_sub_total = civ.negative_rate_in_sub_total

class DisInfoValue(object):
    '''
    A Class for the storage of discrete variables transformation information
    '''
    def __init__(self):
        self.var_name = None
        self.origin_value = []
        self.woe_before = []

def change_feature_dtype(df,variable_type):

    '''
    change feature data type by the variable_type DataFrame
    :param df: dataset DataFrame
    :param variable_type: the DataFrame about variables dtypes
    :return: None
    '''
    s = 'Changing Feature Dtypes'
    print(s.center(60,'-'))
    for vname in df.columns:
        try:
            df[vname] = df[vname].astype(variable_type.loc[vname,'v_type'])
            print(vname,' '*(40-len(vname)),'{0: >10}'.format(variable_type.loc[vname,'v_type']))
        except Exception:
            print('[error]',vname)
            print('[original dtype] ',df.dtypes[vname],' [astype] ',variable_type.loc[vname,'v_type'])
            print('[unique value]',np.unique(df[vname]))

    s = 'Variable Dtypes Have Been Specified'
    print(s.center(60,'-'))
    return

def check_point(df, var, split,min_sample):
    new_split=[]
    if split is not None and split.__len__()>0:
        new_split.append(split[0])
        pdf=df[df[var]<=split[0]]
        if (pdf.shape[0]<min_sample) or (len(np.unique(pdf['target']))<=1):
            new_split.pop()
        for i in range(0,split.__len__()-1):
            new_split1=[float('-inf')]
            for j in range(new_split.__len__()):
                if new_split[j]<split[i]:
                    new_split1.append(new_split[j])
                else:
                    pass
            pdf = df[(df[var] > new_split1[-1]) & (df[var] <= split[i+1])]
            if (pdf.shape[0] < min_sample) or (np.unique(pdf['target']).__len__()<=1):
                continue
            else:
                new_split.append(split[i+1])
        if new_split.__len__()==0:
            pass
        elif ((len(df[df[var] >= new_split[new_split.__len__()-1]])<min_sample)or (np.unique(df[df[var] >= new_split[new_split.__len__()-1]]['target']).__len__()<=1)) :            
            new_split.pop()
        else :
            pass
    else:
        pass
global_bt    return new_split   


def calulate_iv(df,var,global_bt,global_gt):
    '''
    calculate the iv and woe value without split
    :param df:
    :param var:
    :param global_bt:
    :param global_gt:
    :return:
    '''
    # a = df.groupby(['target']).count()
    groupdetail = {}
    bt_sub = sum(df['target'])
    bri = (bt_sub + 0.0001)* 1.0 / bt_sub
    gt_sub = df.shape[0] - bt_sub
    gri = (gt_sub + 0.0001)* 1.0 / gt_sub
    groupdetail['woei'] = np.log(bri / gri)
    groupdetail['ivi'] = (bri - gri) * np.log(bri / gri)
    groupdetail['sub_total_num_percentage'] = df.shape[0]*1.0/(global_bt+global_gt)
    groupdetail['positive_sample_num'] = bt_sub
    groupdetail['negative_sample_num'] = gt_sub
    groupdetail['positive_rate_in_sub_total'] = bt_sub*1.0/df.shape[0]
    groupdetail['negative_rate_in_sub_total'] = gt_sub*1.0/df.shape[0]
    return groupdetail


def calculate_iv_split(df,var,split_point,global_bt,global_gt):
    """
    calculate the iv value with the specified split point
    note:
        the dataset should have variables:'target' which to be encapsulated if have time
    :return:
    """
    #split dataset
    dataset_r = df[df.loc[:,var] > split_point][[var,'target']]
    dataset_l = df[df.loc[:,var] <= split_point][[var,'target']]
    r1_cnt = sum(dataset_r['target'])
    r0_cnt = dataset_r.shape[0] - r1_cnt
    l1_cnt = sum(dataset_l['target'])
    l0_cnt = dataset_l.shape[0] - l1_cnt
    if r0_cnt == 0 or r1_cnt == 0 or l0_cnt == 0 or l1_cnt ==0:
        return 0,0,0,dataset_l,dataset_r,0,0
    lbr = (l1_cnt+ 0.0001)*1.0/
    lgr = (l0_cnt+ 0.0001)*1.0/gt_total
    woel = np.log(lbr/lgr)
    ivl = (lbr-lgr)*woel
    rbr = (r1_cnt+ 0.0001)*1.0/bt_total
    rgr = (r0_cnt+ 0.0001)*1.0/gt_total
    woer = np.log(rbr/rgr)
    ivr = (rbr-rgr)*woer
    iv = ivl+ivr
    return woel,woer,iv,dataset_l,dataset_r,ivl,ivr


def binning_data_split(df,var,global_bt,global_gt,min_sample,way=3,alpha=0.01,bin=5):
    iv_var = InfoValue()
    gd = calulate_iv(df, var,global_bt,global_gt)
    woei, ivi = gd['woei'],gd['ivi']
    
    if way==1:
        split = list(np.unique(np.percentile(df[var],[(i+1)*100/bin for i in range(bin-1)])))
        split.sort()
        split = check_point(df, var, split, min_sample)
        split.sort()
        ivi=calculate_iv_split(df,var,split,global_bt,global_gt)
        iv_var.split_list = split
        iv_var.iv=ivi
        return node(var_name=var,split_point=split,iv=ivi)
    elif way==2:
        split_descend_idx = np.argsort(df[var])
        start_idx = 0
        split=[df.loc[split_descend_idx[start_idx+(i+1)*int(df.shape[0]/bin)-1],var] for i in range(bin-1)]
        split = check_point(df, var, split, min_sample)
        split.sort()
        ivi=calculate_iv_split(df,var,split,global_bt,global_gt)
        iv_var.split_list = split
        iv_var.iv=ivi
        return node(var_name=var,split_point=split,iv=ivi)
    elif way==3:
        percent_value=list(np.unique(np.percentile(df[var], range(100))))
        percent_value.sort()
    
        bestSplit_iv = 0
        bestSplit_woel = []
        bestSplit_woer = []
        bestSplit_ivl = 0
        bestSplit_ivr = 0
        bestSplit_point = []
    
        for point in percent_value[0:percent_value.__len__()-1]:
            if set(df[df[var] > point]['target']).__len__() == 1 or set(df[df[var] <= point]['target']).__len__() == 1 or df[df[var] > point].shape[0] < min_sample or df[df[var] <= point].shape[0] < min_sample :
                continue
            woel, woer, iv, dataset_l, dataset_r, ivl, ivr = calculate_iv_split(df,var,point,global_bt,global_gt)
            if iv > bestSplit_iv:
                bestSplit_woel = woel
                bestSplit_woer = woer
                bestSplit_iv = iv
                bestSplit_point = point
                bestSplit_dataset_r = dataset_r
                bestSplit_dataset_l = dataset_l
                bestSplit_ivl = ivl
                bestSplit_ivr = ivr
        if bestSplit_iv > ivi*(1+alpha) and bestSplit_dataset_r.shape[0] > min_sample and bestSplit_dataset_l.shape[0] > min_sample:
            presplit_right = node()
            presplit_left = node()
            if bestSplit_dataset_r.shape[0] < min_sample or set(bestSplit_dataset_r['target']).__len__() == 1:
                presplit_right.iv = bestSplit_ivr
                right = presplit_right
            else:
                right = binning_data_split(bestSplit_dataset_r,var,global_bt,global_gt,min_sample,way=3,alpha=0.01,bin=5)


            if bestSplit_dataset_l.shape[0] < min_sample or np.unique(bestSplit_dataset_l['target']).__len__() == 1:
                presplit_left.iv = bestSplit_ivl
                left = presplit_left
            else:
                left = binning_data_split(bestSplit_dataset_l,var,global_bt,global_gt,min_sample,way=3,alpha=0.01,bin=5)

            return node(var_name=var,split_point=bestSplit_point,iv=ivi,left=left,right=right)
        else:
            return node(var_name=var,iv=ivi)



def search(tree,split_list):
    '''
    search the tree node
    :param tree: a instance of Tree Node Class
    :return: split points list
    '''
    if isinstance(tree.split_point, list):
        split_list.extend(tree.split_point)
    else:
        split_list.append(tree.split_point)

    if tree.left is not None:
        search(tree.left,split_list)
    if tree.right is not None:
        search(tree.right,split_list)
    return split_list

def format_iv_split(df,var,split_list,global_bt,global_gt):
    '''
    Given the dataset DataFrame and split points list then return a InfoValue instance;
    Just for continuous variable
    :param df:
    :param var:
    :param split_list:
    :param global_bt:
    :param global_gt:
    :return:
    '''
    civ = InfoValue()
    civ.var_name = var
    civ.split_list = split_list
    dfcp = df[:]
    civ.sub_total_sample_num = []
    civ.positive_sample_num = []
    civ.negative_sample_num = []
    civ.sub_total_num_percentage = []
    civ.positive_rate_in_sub_total = []

    for i in range(0, split_list.__len__()):
        dfi = dfcp[dfcp[var] <= split_list[i]]
        dfcp = dfcp[dfcp[var] > split_list[i]]
        gd = calulate_iv(dfi, var,global_bt,global_gt)
        woei, ivi = gd['woei'],gd['ivi']
        civ.woe_list.append(woei)
        civ.iv_list.append(ivi)
        civ.sub_total_sample_num.append(dfi.shape[0])
        civ.positive_sample_num.append(gd['positive_sample_num'])
        civ.negative_sample_num.append(gd['negative_sample_num'])
        civ.sub_total_num_percentage.append(gd['sub_total_num_percentage'])
        civ.positive_rate_in_sub_total.append(gd['positive_rate_in_sub_total'])
        civ.negative_rate_in_sub_total.append(gd['negative_rate_in_sub_total'])

    if dfcp.shape[0]>0:
        gd = calulate_iv(dfcp, var,global_bt,global_gt)
        woei, ivi = gd['woei'],gd['ivi']
        civ.woe_list.append(woei)
        civ.iv_list.append(ivi)
        civ.sub_total_sample_num.append(dfcp.shape[0])
        civ.positive_sample_num.append(gd['positive_sample_num'])
        civ.negative_sample_num.append(gd['negative_sample_num'])
        civ.sub_total_num_percentage.append(gd['sub_total_num_percentage'])
        civ.positive_rate_in_sub_total.append(gd['positive_rate_in_sub_total'])
        civ.negative_rate_in_sub_total.append(gd['negative_rate_in_sub_total'])
    civ.iv = sum(civ.iv_list)
    return civ





def woe_trans(dvar,civ):
    # replace the var value with the given woe value
    var = copy.deepcopy(dvar)
    if not civ.is_discrete:
        if civ.woe_list.__len__()>1:
            split_list = []
            split_list.append(float("-inf"))
            split_list.extend([i for i in civ.split_list])
            split_list.append(float("inf"))
            for i in range(civ.woe_list.__len__()):
                var[(dvar > split_list[i]) & (dvar <= split_list[i+1])] = civ.woe_list[i]
        else:
            var[:] = civ.woe_list[0]
    else:
        split_map = {}
        for i in range(civ.split_list.__len__()):
            for j in range(civ.split_list[i].__len__()):
                split_map[civ.split_list[i][j]] = civ.woe_list[i]
        var = var.map(split_map)
    return var



def proc_woe_discrete(df,var,global_bt,global_gt,min_sample,way,alpha,bin):
    '''
    process woe transformation of discrete variables
    :param df:
    :param var:
    :param global_bt:
    :param global_gt:
    :param min_sample:
    :return:
    '''
    s = 'process discrete variable:'+str(var)
    print(s.center(60, '-'))
    df = df[[var,'target']]
    div = DisInfoValue()
    div.var_name = var
    rdict = {}
    cpvar = df[var]
    # print('np.unique(df[var])：',np.unique(df[var]))
    for var_value in np.unique(df[var]):
        # Here come with a '==',in case type error you must do Nan filling process firstly
        df_temp = df[df[var] == var_value]
        bad_cnt=sum(df_temp['target'])
        good_cnt=df_temp.shape[0]-bad_cnt
        lbr = (bad_cnt+ 0.0001)*1.0/global_bt
        lgr = (good_cnt+ 0.0001)*1.0/global_gt
        woei = np.log(lbr/lgr)
        ivi = (lbr-lgr)*woei
        div.origin_value.append(var_value)
        div.woe_before.append(woei)
        rdict[var_value] = woei
        # print(var_value,woei,ivi)
    cpvar = cpvar.map(rdict)
    df[var] = cpvar
    iv_tree = binning_data_split(df,var,global_bt,global_gt,min_sample,way,alpha,bin)
    # Traversal tree, get the segmentation point
    split_list = []
    search(iv_tree, split_list)
    split_list = list(np.unique([1.0 * x for x in split_list if x is not None]))
    split_list.sort()
    # Segmentation point checking and processing
    split_list = check_point(df, var, split_list, min_sample)
    split_list.sort()
    civ = format_iv_split(df, var, split_list,global_bt,global_gt)
    civ.is_discrete = 1
    split_list_temp = []
    split_list_temp.append(float("-inf"))
    split_list_temp.extend([i for i in split_list])
    split_list_temp.append(float("inf"))

    a = []
    for i in range(split_list_temp.__len__() - 1):
        temp = []
        for j in range(div.origin_value.__len__()):
            if (div.woe_before[j]>split_list_temp[i]) & (div.woe_before[j]<=split_list_temp[i+1]):
                temp.append(div.origin_value[j])
        if temp != [] :
            a.append(temp)
    civ.split_list = a
    return civ


def proc_woe_continuous(df,var,global_bt,global_gt,min_sample,way,alpha,bin):
    '''
    process woe transformation of discrete variables
    :param df:
    :param var:
    :param global_bt:
    :param global_gt:
    :param min_sample:
    :return:
    '''
    s = 'process continuous variable:'+str(var)
    print(s.center(60, '-'))
    df = df[[var,'target']]
    iv_tree = binning_data_split(df, var,global_bt,global_gt,min_sample,way,alpha,bin)
    # Traversal tree, get the segmentation point
    split_list = []
    search(iv_tree, split_list)
    split_list = list(np.unique([1.0 * x for x in split_list if x is not None]))
    split_list.sort()

    # Segmentation point checking and processing
    split_list = check_point(df, var, split_list, min_sample)
    split_list.sort()
    civ = format_iv_split(df, var,split_list,global_bt,global_gt)
    return civ


##########################添加最优分组和等频、等宽的选择参数
def process_train_woe(infile_path=None,outfile_path=None,rst_path=None,config_path=None,way=3,alpha=0.05,bin=5):
    print('run into process_train_woe: \n',time.asctime(time.localtime(time.time())))
    data_path = infile_path
    cfg = config.config()
    cfg.load_file(config_path,data_path)
    bin_var_list = [tmp for tmp in cfg.bin_var_list if tmp in list(cfg.dataset_train.columns)]
    for var in bin_var_list:
        # fill null
        cfg.dataset_train.loc[cfg.dataset_train[var].isnull(), (var)] = -1
    # change feature dtypes
    change_feature_dtype(cfg.dataset_train, cfg.variable_type)
    rst = []
    # process woe transformation of continuous variables
    print('process woe transformation of continuous variables: \n',time.asctime(time.localtime(time.time())))
    print('cfg.global_bt',cfg.global_bt)
    print('cfg.global_gt', cfg.global_gt)
    for var in bin_var_list:
        rst.append(proc_woe_continuous(cfg.dataset_train,var,cfg.global_bt,cfg.global_gt,cfg.min_sample,way,alpha,bin))          
    # process woe transformation of discrete variables
    print('process woe transformation of discrete variables: \n',time.asctime(time.localtime(time.time())))
    for var in [tmp for tmp in cfg.discrete_var_list if tmp in list(cfg.dataset_train.columns)]:
        # fill null
        cfg.dataset_train.loc[cfg.dataset_train[var].isnull(), (var)] = 'missing'
        rst.append(proc_woe_discrete(cfg.dataset_train,var,cfg.global_bt,cfg.global_gt,cfg.min_sample,way,alpha,bin))
    feature_detail = eval.eval_feature_detail(rst, outfile_path)
    print('save woe transformation rule into pickle: \n',time.asctime(time.localtime(time.time())))
    output = open(rst_path, 'wb')
    pickle.dump(rst,output)
    output.close()
    return feature_detail,rst

#########################################转之前要对数据做缺失值等数据处理
def process_woe_trans(in_data_path=None,rst_path=None,out_path=None,config_path=None):
    cfg = config.config()
    cfg.load_file(config_path, in_data_path)
    for var in [tmp for tmp in cfg.bin_var_list if tmp in list(cfg.dataset_train.columns)]:
        # fill null
        cfg.dataset_train.loc[cfg.dataset_train[var].isnull(), (var)] = -1
    for var in [tmp for tmp in cfg.discrete_var_list if tmp in list(cfg.dataset_train.columns)]:
        # fill null
        cfg.dataset_train.loc[cfg.dataset_train[var].isnull(), (var)] = 'missing'
    change_feature_dtype(cfg.dataset_train, cfg.variable_type)
    output = open(rst_path, 'rb')
    rst = pickle.load(output)
    output.close()
    # Training dataset Woe Transformation
    for r in rst:
        if r.var_name in list(cfg.dataset_train.columns):
            cfg.dataset_train[r.var_name] = woe_trans(cfg.dataset_train[r.var_name], r)
    cfg.dataset_train.to_csv(out_path,index=False)
    
######################################填充缺失值 
def fill_na(dataset,candidate_var_list,discrete_var_list,discrete_filler='missing',way='constant',continuous_filler=-1):
    for var in [tmp for tmp in discrete_var_list if tmp in list(dataset.columns)]:
        dataset.loc[dataset[var].isnull(), (var)] = discrete_filler
    for var in [tmp for tmp in candidate_var_list if tmp in list(dataset.columns)]:
        if way=='constant':
            dataset.loc[dataset[var].isnull(), (var)] = continuous_filler
        elif way=='mean':
            dataset.loc[dataset[var].isnull(), (var)] = dataset[var].mean()
        elif way=='median':
            dataset.loc[dataset[var].isnull(), (var)] = dataset[var].median()
        elif way=='mode':
            dataset.loc[dataset[var].isnull(), (var)] = dataset[var].mode()
        elif way=='lagrange':
            def ploy(s,n,k=6):
                y=s[list(range(n-k,n))+list(range(n+1,n+1+k))]
                if len(y[y.notnull()])==0:
                    return s.median()
                else:
                    y=y[y.notnull()]
                return lagrange(y.index,list(y))(n)
            for j in range(len(dataset[var])):
                if (dataset[var].isnull())[j]:
                    dataset[var][j]=ploy(dataset[var],j)
                    
