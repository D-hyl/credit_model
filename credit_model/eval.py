# -*- coding: utf-8 -*-
"""
Created on Mon Apr 09 16:43:43 2018

@author: yingliang.huang
"""

import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
import credit_model.config as config
import pickle
from matplotlib.ticker import MultipleLocator,FormatStrFormatter
from sklearn.metrics import roc_curve, auc


def compute_ks(target,proba):
    '''
    target: numpy array of shape (1,)
    proba: numpy array of shape (1,), predicted probability of the sample being positive
    returns:
    ks: float, ks score estimation
    '''
    get_ks = lambda proba, target: ks_2samp(proba[target == 1], proba[target != 1]).statistic
    return get_ks(proba, target)



def eval_feature_detail(Info_Value_list,out_path=False):
    """
    format InfoValue list to Dataframe
    :param Info_Value_list: Instance list of Class InfoValue
    :param out_path:specify the Dataframe to csv file path ,default False
    :return:DataFrame about feature detail
    """
    rst = Info_Value_list
    format_rst = []
    for kk in range(0,len(rst)):
        print(rst[kk].var_name)
        split_list = []
        if rst[kk].split_list != []:
            if not rst[kk].is_discrete:
                #deal with split_list
                split_list.append('(-INF,'+str(rst[kk].split_list[0])+']')
                for i in range(0,len(rst[kk].split_list)-1):
                    split_list.append('(' + str(rst[kk].split_list[i])+','+ str(rst[kk].split_list[i+1]) + ']')
                split_list.append('(' + str(rst[kk].split_list[len(rst[kk].split_list)-1]) + ',+INF)')
            else:
                split_list = rst[kk].split_list
        else:
            split_list.append('(-INF,+INF)')

        # merge into dataframe
        columns = ['var_name','split_list','sub_total_sample_num','positive_sample_num'
            ,'negative_sample_num','sub_total_num_percentage','positive_rate_in_sub_total'
            ,'woe_list','iv_list','iv']
        rowcnt = len(rst[kk].iv_list)
        if rowcnt < len(split_list):
            split_list = split_list[:rowcnt]
        var_name = [rst[kk].var_name] * rowcnt
        iv = [rst[kk].iv] * rowcnt
        iv_list = rst[kk].iv_list
        woe_list = rst[kk].woe_list
        a = pd.DataFrame({'var_name':var_name,'iv_list':iv_list,'woe_list':woe_list
                             ,'split_list':split_list,'iv':iv,'sub_total_sample_num':rst[kk].sub_total_sample_num
                             ,'positive_sample_num':rst[kk].positive_sample_num,'negative_sample_num':rst[kk].negative_sample_num
                             ,'sub_total_num_percentage':rst[kk].sub_total_num_percentage
                             ,'positive_rate_in_sub_total':rst[kk].positive_rate_in_sub_total
                             ,'negative_rate_in_sub_total':rst[kk].negative_rate_in_sub_total},columns=columns)
        format_rst.append(a)
    # merge dataframe list into one dataframe vertically
    cformat_rst = pd.concat(format_rst)
    if out_path:
        file_name = out_path if isinstance(out_path, str) else None
        cformat_rst.to_csv(file_name, index=False,encoding='utf-8')
    return cformat_rst

def wald_test(model,X):
    '''
    :param model: a model file that should have predict_proba() function
    :param X: dataset features DataFrame
    :return: the value of wald_stats,p_value
    '''
    pred_probs = np.matrix(model.predict_proba(X))
    X_design = np.hstack((np.ones(shape=(X.shape[0], 1)), X))
    diag_array = np.multiply(pred_probs[:, 0], pred_probs[:, 1]).A1
    V = scipy.sparse.diags(diag_array)
    m1 = X_design.T * V
    m2 = m1.dot(X_design)
    cov_mat = np.linalg.inv(m2)
    model_params = np.hstack((model.intercept_[0], model.coef_[0]))
    wald_stats = (model_params / np.sqrt(np.diag(cov_mat))) ** 2
    wald = scipy.stats.wald()
    p_value = wald.pdf(wald_stats)
    return wald_stats,p_value



def eval_model_stability(proba_train, proba_validation, segment_cnt = 10,out_path=False):

    step = 1.0/segment_cnt
    flag = 0.0
    model_stability = []
    len_train = len(proba_train)
    len_validation = len(proba_validation)
    columns = ['score_range','segment_train_percentage','segment_validation_percentage','difference',
               'variance','ln_variance','stability_index']

    while flag < 1.0:
        temp = {}
        score_range = '['+str(flag)+','+str(flag + step)+')'
        segment_train_cnt = proba_train[(proba_train >= flag) & (proba_train < flag + step)].count()
        segment_train_percentage = segment_train_cnt*1.0/len_train
        segment_validation_cnt = proba_validation[(proba_validation >= flag) & (proba_validation < flag + step)].count()
        segment_validation_percentage = segment_validation_cnt * 1.0 / len_validation
        difference = segment_validation_percentage - segment_train_percentage
        variance = float(segment_validation_percentage)/segment_train_percentage
        ln_variance = np.log(variance)
        stability_index = difference * ln_variance

        temp['score_range'] = score_range
        temp['segment_train_percentage'] = segment_train_percentage
        temp['segment_validation_percentage'] = segment_validation_percentage
        temp['difference'] = difference
        temp['variance'] = variance
        temp['ln_variance'] = ln_variance
        temp['stability_index'] = stability_index
        model_stability.append(temp)
        flag += step

    model_stability = pd.DataFrame(model_stability,columns=columns)

    if out_path:
        file_name = out_path if isinstance(out_path, str) else None
        model_stability.to_csv(file_name, index=False)
    return model_stability,model_stability['stability_index'].sum()


def eval_feature_stability(civ_list, df_train, df_validation,candidate_var_list,out_path=False):

    psi_dict = {}
    civ_var_list = [civ_list[i].var_name for i in range(len(civ_list))]
    intersection = list(set(civ_var_list).intersection(set(candidate_var_list)))
    civ_idx_list = [civ_var_list.index(var) for var in intersection]
    len_train = len(df_train)
    len_validation = len(df_validation)
    df_train.columns=[var.split('.')[-1] for var in df_train.columns]
    df_validation.columns=[var.split('.')[-1] for var in df_validation.columns]
    psi_dict['feature_name'] = []
    psi_dict['group'] = []
    psi_dict['segment_train_cnt'] = []
    psi_dict['segment_train_percentage'] = []
    psi_dict['segment_validation_cnt'] = []
    psi_dict['segment_validation_percentage'] = []
    for i in civ_idx_list:
        if civ_list[i].is_discrete:
            for j in range(len(civ_list[i].split_list)):
                psi_dict['feature_name'].append(civ_list[i].var_name)
                psi_dict['group'].append(civ_list[i].split_list[j])
                civ_split_list = civ_list[i].split_list[j]
                segment_train_cnt = 0
                for m in civ_split_list:
                    segment_train_cnt += df_train[civ_list[i].var_name][df_train[civ_list[i].var_name] == m].count()
                psi_dict['segment_train_cnt'].append(segment_train_cnt)
                psi_dict['segment_train_percentage'].append(float(segment_train_cnt)/len_train)
                segment_validation_cnt = 0
                for m in civ_split_list:
                    segment_validation_cnt += df_validation[civ_list[i].var_name][df_validation[civ_list[i].var_name] == m].count()
                psi_dict['segment_validation_cnt'].append(segment_validation_cnt)
                psi_dict['segment_validation_percentage'].append(float(segment_validation_cnt)/len_validation)

        else:
            split_list = []
            split_list.append(float("-inf"))
            split_list.extend([temp for temp in civ_list[i].split_list])
            split_list.append(float("inf"))
            var_name = civ_list[i].var_name

            for j in range(len(split_list)-2):
                psi_dict['feature_name'].append(civ_list[i].var_name)
                psi_dict['group'].append('('+str(split_list[j])+','+str(split_list[j+1])+']')
                segment_train_cnt = df_train[var_name][(df_train[var_name] > split_list[j])&(df_train[var_name] <= split_list[j+1])].count()
                psi_dict['segment_train_cnt'].append(segment_train_cnt)
                psi_dict['segment_train_percentage'].append(float(segment_train_cnt)/len_train)
                segment_validation_cnt = df_validation[var_name][(df_validation[var_name] > split_list[j])&(df_validation[var_name] <= split_list[j+1])].count()
                psi_dict['segment_validation_cnt'].append(segment_validation_cnt)
                psi_dict['segment_validation_percentage'].append(float(segment_validation_cnt)/len_validation)
            psi_dict['feature_name'].append(var_name)
            psi_dict['group'].append('(' + str(split_list[len(split_list)-2]) + ',+INF)')
            segment_train_cnt = df_train[var_name][df_train[var_name] > split_list[len(split_list)-2]].count()
            psi_dict['segment_train_cnt'].append(segment_train_cnt)
            psi_dict['segment_train_percentage'].append(float(segment_train_cnt) / len_train)
            segment_validation_cnt = df_validation[var_name][df_validation[var_name] > split_list[len(split_list)-2]].count()
            psi_dict['segment_validation_cnt'].append(segment_validation_cnt)
            psi_dict['segment_validation_percentage'].append(float(segment_validation_cnt) / len_validation)

    psi_dict['difference'] = pd.Series(psi_dict['segment_validation_percentage']) - pd.Series(psi_dict['segment_train_percentage'])
    psi_dict['variance'] = list(map(lambda x_y: x_y[0] / (x_y[1]+0.000000001), zip(psi_dict['segment_validation_percentage'], psi_dict['segment_train_percentage'])))
    psi_dict['Ln(variance)'] = np.log(np.array(psi_dict['variance'])+0.000000001)
    psi_dict['stability_index'] = np.array(psi_dict['difference']) * np.array(psi_dict['Ln(variance)'])
    columns = ['feature_name','group','segment_train_cnt','segment_train_percentage',
               'segment_validation_cnt','segment_validation_percentage','difference',
               'variance','Ln(variance)','stability_index']

    psi_df = pd.DataFrame(psi_dict, columns=columns)
    result = psi_df.groupby('feature_name').sum()
    result.reset_index(inplace=True)
    if out_path:
        file_name = out_path if isinstance(out_path, str) else None
        psi_df.to_csv(file_name, index=False)
    return psi_df,result[['feature_name','stability_index']]



def lift(target, predict_proba,  segment_cnt = 20,out_path=False):

    proba_descend_idx = np.argsort(predict_proba)
    proba_descend_idx = proba_descend_idx[::-1]

    grp_idx = 1
    start_idx = 0
    total_sample_cnt = len(predict_proba)
    total_positive_sample_cnt = target.sum()
    total_positive_sample_percentage = total_positive_sample_cnt/total_sample_cnt
    segment_sample_cnt = int(len(predict_proba) / segment_cnt)

    segment_list = []
    columns = ['grp_idx', 'segment_sample_cnt', 'segment_pos_cnt', 'segment_pos_percent',
               'total_pos_percent', 'lift']

    while start_idx < total_sample_cnt:
        s = {}
        s['grp_idx'] = grp_idx
        segment_idx_list = proba_descend_idx[start_idx : start_idx + segment_sample_cnt]
        segment_target = target[segment_idx_list]
        segment_sample_cnt = len(segment_idx_list)
        s['segment_sample_cnt'] = segment_sample_cnt
        segment_pos_cnt = segment_target.sum()
        s['segment_pos_cnt'] = segment_pos_cnt
        segment_pos_percent = segment_pos_cnt/segment_sample_cnt
        s['segment_pos_percent'] = segment_pos_percent
        s['total_pos_percent'] = total_positive_sample_percentage 
        lift=round(segment_pos_percent/total_positive_sample_percentage,2)
        s['lift'] = lift 

        segment_list.append(s)
        grp_idx += 1
        start_idx += segment_sample_cnt
    segment_list = pd.DataFrame(segment_list,columns=columns)
    segment_list.drop(segment_list.index[-1],inplace=True)
    if out_path:
        file_name = out_path if isinstance(out_path, str) else None
        segment_list.to_csv(file_name, index=False)
    plt.figure()
    plt.plot(segment_list['grp_idx'],segment_list['lift'], label="lift")
    plt.legend()
    plt.xlabel('group')
    plt.ylabel('lift')
    plt.title('Lift of Model')
    ax = plt.subplot(111)
    xmajorLocator=MultipleLocator(2)
    xmajorFormatter=FormatStrFormatter('%d')
    ax.xaxis.set_major_locator(xmajorLocator)
    ax.xaxis.set_major_formatter(xmajorFormatter)
    plt.show()
    return segment_list


def IV_of_feature(civ_list, df,candidate_var_list,out_path=False):

    IV = {}
    civ_var_list = [civ_list[i].var_name for i in range(len(civ_list))]
    intersection = list(set(civ_var_list).intersection(set(candidate_var_list)))
    civ_idx_list = [civ_var_list.index(var) for var in intersection]
    len_df = len(df)
    bad_total=df['target'].sum()
    good_total=len_df-bad_total
    df.columns=[var.split('.')[-1] for var in df.columns]
    IV['feature_name'] = []
    IV['group'] = []
    IV['segment_bad_cnt'] = []
    IV['segment_bad_percentage'] = []
    IV['segment_good_cnt'] = []
    IV['segment_good_percentage'] = []
    IV['total']=[]
    IV['badrate']=[]
    IV['percent']=[]
    for i in civ_idx_list:
        if civ_list[i].is_discrete:
            for j in range(len(civ_list[i].split_list)):
                IV['feature_name'].append(civ_list[i].var_name)
                IV['group'].append(civ_list[i].split_list[j])
                civ_split_list = civ_list[i].split_list[j]
                segment_bad_cnt = 0
                for m in civ_split_list:
                    segment_bad_cnt += df['target'][df[civ_list[i].var_name] == m].sum()
                IV['segment_bad_cnt'].append(segment_bad_cnt)
                IV['segment_bad_percentage'].append(float(segment_bad_cnt)/bad_total)
                segment_good_cnt = 0
                for m in civ_split_list:
                    segment_good_cnt += df['target'][df[civ_list[i].var_name] == m].count()-df['target'][df[civ_list[i].var_name] == m].sum()
                IV['segment_good_cnt'].append(segment_good_cnt)
                IV['segment_good_percentage'].append(float(segment_good_cnt)/good_total)
                IV['total'].append(segment_bad_cnt+segment_good_cnt)
                IV['badrate'].append(round(float(segment_bad_cnt)/(segment_bad_cnt+segment_good_cnt+0.0001),4))
                IV['percent'].append(round(float(segment_bad_cnt+segment_good_cnt)/len_df,4))
        else:
            split_list = []
            split_list.append(float("-inf"))
            split_list.extend([temp for temp in civ_list[i].split_list])
            split_list.append(float("inf"))
            var_name = civ_list[i].var_name

            for j in range(len(split_list)-2):
                IV['feature_name'].append(civ_list[i].var_name)
                IV['group'].append('('+str(split_list[j])+','+str(split_list[j+1])+']')
                segment_bad_cnt = df['target'][(df[var_name] > split_list[j])&(df[var_name] <= split_list[j+1])].sum()
                IV['segment_bad_cnt'].append(segment_bad_cnt)
                IV['segment_bad_percentage'].append(float(segment_bad_cnt)/bad_total)
                segment_good_cnt = df['target'][(df[var_name] > split_list[j])&(df[var_name] <= split_list[j+1])].count()-df['target'][(df[var_name] > split_list[j])&(df[var_name] <= split_list[j+1])].sum()
                IV['segment_good_cnt'].append(segment_good_cnt)
                IV['segment_good_percentage'].append(float(segment_good_cnt)/good_total)
                IV['total'].append(segment_bad_cnt+segment_good_cnt)
                IV['badrate'].append(round(float(segment_bad_cnt)/(segment_bad_cnt+segment_good_cnt+0.0001),4))
                IV['percent'].append(round(float(segment_bad_cnt+segment_good_cnt)/len_df,4))
            IV['feature_name'].append(var_name)
            IV['group'].append('(' + str(split_list[len(split_list)-2]) + ',+INF)')
            segment_bad_cnt = df['target'][df[var_name] > split_list[len(split_list)-2]].sum()
            IV['segment_bad_cnt'].append(segment_bad_cnt)
            IV['segment_bad_percentage'].append(float(segment_bad_cnt) / bad_total)
            segment_good_cnt = df['target'][df[var_name] > split_list[len(split_list)-2]].count()-df['target'][df[var_name] > split_list[len(split_list)-2]].sum()
            IV['segment_good_cnt'].append(segment_good_cnt)
            IV['segment_good_percentage'].append(float(segment_good_cnt) / good_total)
            IV['total'].append(segment_bad_cnt+segment_good_cnt)
            IV['badrate'].append(round(float(segment_bad_cnt)/(segment_bad_cnt+segment_good_cnt+0.0001),4))
            IV['percent'].append(round(float(segment_bad_cnt+segment_good_cnt)/len_df,4))

    IV['difference'] = pd.Series(IV['segment_bad_percentage']) - pd.Series(IV['segment_good_percentage'])
    IV['variance'] = list(map(lambda x_y: x_y[0] / (x_y[1]+0.000000001), zip(IV['segment_bad_percentage'], IV['segment_good_percentage'])))
    IV['woe'] = np.log(np.array(IV['variance'])+0.000000001)
    IV['iv'] = np.array(IV['difference']) * np.array(IV['woe'])
    columns = ['feature_name','group','segment_bad_cnt','segment_bad_percentage',
               'segment_good_cnt','segment_good_percentage','total','badrate','percent','difference',
               'variance','woe','iv']
    iv_df = pd.DataFrame(IV, columns=columns)
    result = iv_df.groupby('feature_name').sum()
    result.reset_index(inplace=True)
    if out_path:
        file_name = out_path if isinstance(out_path, str) else None
        iv_df.to_csv(file_name, index=False)
    return iv_df,result[['feature_name','iv']]


def AUC_of_Module(target,p,out_path=False):
    fpr,tpr,threshold = roc_curve(target,p)
    roc_auc = auc(fpr,tpr)
    plt.figure(figsize=(10,10))
    lw=2
    plt.plot(fpr,tpr,color='darkorange',lw=lw,label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0,1],[0,1],color='navy',lw=lw,linestyle='--')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.title('ROC')
    
    if out_path:
        file_name = out_path if isinstance(out_path, str) else None
        plt.savefig(file_name)
    else:
        plt.show()
        
    return roc_auc



def plot_ks(target,proba,axistype='pct',out_path=False):
    """
    plot k-s figure
    :param proba: 1-d array,prediction probability values
    :param target: 1-d array,the list of actual target value
    :param axistype: specify x axis :'axistype' must be either 'pct' (sample percent) or 'proba' (prediction probability)
    :param out_path: specify the file path to store ks plot figure,default False
    :return: DataFrame, figure summary
    """
    assert axistype in ['pct','proba'] , "KS Plot TypeError: Attribute 'axistype' must be either 'pct' or 'proba' !"
    a = pd.DataFrame(np.array([proba,target]).T,columns=['proba','target'])
    a.sort_values(by='proba',ascending=False,inplace=True)
    a['sum_Times']=a['target'].cumsum()
    total_1 = a['target'].sum()
    total_0 = len(a) - a['target'].sum()

    a['temp'] = 1
    a['Times']=a['temp'].cumsum()
    a['cdf1'] = a['sum_Times']/total_1
    a['cdf0'] = (a['Times'] - a['sum_Times'])/total_0
    a['ks'] = a['cdf1'] - a['cdf0']
    a['percent'] = a['Times']*1.0/len(a)

    idx = np.argmax(a['ks'])
    # print(a.loc[idx])
    if axistype == 'pct':
        '''
        KS曲线,横轴为按照输出的概率值排序后的观察样本比例
        '''
        plt.figure()
        plt.plot(a['percent'],a['cdf1'], label="CDF_positive")
        plt.plot(a['percent'],a['cdf0'],label="CDF_negative")
        plt.plot(a['percent'],a['ks'],label="K-S")
        sx = np.linspace(0,1,10)
        sy = sx
        plt.plot(sx,sy,linestyle='--',color='darkgrey',linewidth=1.2)
        plt.legend(loc='NorthWest')
        plt.grid(True)
        ymin, ymax = plt.ylim()
        plt.xlabel('Sample percent')
        plt.ylabel('Cumulative probability')
        plt.title('Model Evaluation Index K-S')
        plt.axis('tight')

        # 虚线
        t = a.loc[idx]['percent']
        yb = round(a.loc[idx]['cdf1'],4)
        yg = round(a.loc[idx]['cdf0'],4)
        plt.plot([t,t],[yb,yg], color ='red', linewidth=1.4, linestyle="--")
        plt.scatter([t,],[yb,], 20, color ='dodgerblue')
        plt.annotate(r'$recall_p=%s$' % round(a.loc[idx]['cdf1'],4), xy=(t, yb), xycoords='data', xytext=(+10, -5),
                     textcoords='offset points', fontsize=8,
                     arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.1"))

        plt.scatter([t,],[yg,], 20, color ='darkorange')
        plt.annotate(r'$recall_n=%s$' % round(a.loc[idx]['cdf0'],4), xy=(t, yg), xycoords='data', xytext=(+10, -10),
                     textcoords='offset points', fontsize=8,
                     arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.1"))
        # K-S曲线峰值
        plt.scatter([t,],[a.loc[idx]['ks'],], 20, color ='limegreen')
        plt.annotate(r'$ks=%s,p=%s$' % (round(a.loc[idx]['ks'],4)
                                        ,round(a.loc[idx]['proba'],4))
                     , xy=(a.loc[idx]['percent'], a.loc[idx]['ks'])
                     , xycoords='data'
                     , xytext=(+15, -15),
                     textcoords='offset points'
                     , fontsize=8
                     ,arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.1"))
        plt.annotate(r'$percent=%s,cnt=%s$' % (round(a.loc[idx]['percent'],4)
                                               ,round(a.loc[idx]['Times'],0))
                     , xy=(a.loc[idx]['percent'], a.loc[idx]['ks'])
                     , xycoords='data'
                     , xytext=(+25, -25),
                     textcoords='offset points'
                     , fontsize=8
                     )
    else:
        '''
        改变横轴,横轴为模型输出的概率值
        '''
        plt.figure()
        plt.grid(True)
        plt.plot(1-a['proba'],a['cdf1'], label="CDF_bad")
        plt.plot(1-a['proba'],a['cdf0'],label="CDF_good")
        plt.plot(1-a['proba'],a['ks'],label="ks")
        plt.legend()
        ymin, ymax = plt.ylim()
        plt.xlabel('1-[Predicted probability]')
        plt.ylabel('Cumulative probability')
        plt.title('Model Evaluation Index K-S')
        plt.axis('tight')
        plt.show()
        # 虚线
        t = 1 - a.loc[idx]['proba']
        yb = round(a.loc[idx]['cdf1'],4)
        yg = round(a.loc[idx]['cdf0'],4)
        plt.plot([t,t],[yb,yg], color ='red', linewidth=1.4, linestyle="--")
        plt.scatter([t,],[yb,], 20, color ='dodgerblue')
        plt.annotate(r'$recall_p=%s$' % round(a.loc[idx]['cdf1'],4), xy=(t, yb), xycoords='data', xytext=(+10, -5),
                     textcoords='offset points', fontsize=8,
                     arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.1"))
        plt.scatter([t,],[yg,], 20, color ='darkorange')
        plt.annotate(r'$recall_n=%s$' % round(a.loc[idx]['cdf0'],4), xy=(t, yg), xycoords='data', xytext=(+10, -10),
                     textcoords='offset points', fontsize=8,
                     arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.1"))
        # K-S曲线峰值
        plt.scatter([t,],[a.loc[idx]['ks'],], 20, color ='limegreen')
        plt.annotate(r'$ks=%s,p=%s$' % (round(a.loc[idx]['ks'],4)
                                        ,round(a.loc[idx]['proba'],4))
                     , xy=(t, a.loc[idx]['ks'])
                     , xycoords='data'
                     , xytext=(+15, -15),
                     textcoords='offset points'
                     , fontsize=8
                     ,arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.1"))
        plt.annotate(r'$percent=%s,cnt=%s$' % (round(a.loc[idx]['percent'],4)
                                               ,round(a.loc[idx]['Times'],0))
                     , xy=(t, a.loc[idx]['ks'])
                     , xycoords='data'
                     , xytext=(+25, -25),
                     textcoords='offset points'
                     , fontsize=8
                     )

    if out_path:
        file_name = out_path if isinstance(out_path, str) else None
        plt.savefig(file_name)
    else:
        plt.show()
    return a.loc[idx]
