#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def draw_hist_cat(df, arr,**kwargs):
    feature = df[arr][~df[arr].isna()]
    rotation = kwargs.get('rotation', 0)
    color = '#86bf91' if kwargs.get('change_color', None) else '#0076B0'
    
    fig = plt.figure(figsize=(10,6))
    plt.xticks(fontsize=10, rotation = rotation)
    plt.yticks(fontsize=10, rotation = rotation)
    
    n = len(feature.unique())
    feature.hist(bins=range(n+1), grid=False, color = color, zorder=2, rwidth=0.9, align='left')
    title_name = 'Distribution of ' + arr
    plt.title(title_name, fontsize=10)
    plt.show()

def convert_categorical_numerical(df, feature_removeString):
    for feature in feature_removeString.keys():
        for string in feature_removeString[feature]:
            df.loc[:, feature] = df.loc[:, feature].apply(lambda x: str(x).replace(string, '').strip())
        # if empty string after removing characters, fill with 0
        df.loc[df[feature] == '', feature] = 0

def draw_PR(model, dtrain, dvalid, dtest, y_train, y_valid, y_test):
    probas_0 = model.predict(dtrain, ntree_limit=model.best_ntree_limit)
    probas_1 = model.predict(dvalid, ntree_limit=model.best_ntree_limit)
    probas_2 = model.predict(dtest, ntree_limit=model.best_ntree_limit)
    
    # PR curve is more informative than ROC when dealing with highly skewed datasets
    precision_0, recall_0, _ = precision_recall_curve(y_train, probas_0)
    precision_1, recall_1, _ = precision_recall_curve(y_valid, probas_1)
    precision_2, recall_2, _ = precision_recall_curve(y_test, probas_2)
    
    auc_0 = auc(recall_0, precision_0)
    auc_1 = auc(recall_1, precision_1)
    auc_2 = auc(recall_2, precision_2)
    
    print ('Area under the PR curve - train: %f' % auc_0)
    print ('Area under the PR curve - validation: %f' % auc_1)
    print ('Area under the PR curve - test: %f' % auc_2)
    # Plot RP curve
    plt.figure(figsize=(8,8))
    plt.plot(recall_0, precision_0, label='PR curve - train (AUC = %0.2f)' % auc_0, color='b')
    plt.plot(recall_1, precision_1, label='PR curve - valid (AUC = %0.2f)' % auc_1, color='r')
    plt.plot(recall_2, precision_2, label='PR curve - test  (AUC = %0.2f)' % auc_2, color='g')
    no_skill = len(y_test[y_test==1]) / len(y_test)
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR for lead score model')
    plt.legend(loc="upper right")
    plt.show()

def draw_ROC(model, dtrain, dvalid, dtest, y_train, y_valid, y_test ):
    probas_ = model.predict(dvalid, ntree_limit=model.best_ntree_limit)
    probas_1 = model.predict(dtrain, ntree_limit=model.best_ntree_limit)
    probas_2 = model.predict(dtest, ntree_limit=model.best_ntree_limit)
    fpr, tpr, thresholds = roc_curve(y_valid, probas_)
    fpr_1, tpr_1, thresholds_1 = roc_curve(y_train, probas_1)
    fpr_2, tpr_2, thresholds_2 = roc_curve(y_test, probas_2)
    roc_auc = auc(fpr, tpr)
    roc_auc_1 = auc(fpr_1, tpr_1)
    roc_auc_2 = auc(fpr_2, tpr_2)
    print ("Area under the ROC curve - validation: %f" % roc_auc)
    print ("Area under the ROC curve - train: %f" % roc_auc_1)
    print ("Area under the ROC curve - test: %f" % roc_auc_2)
    # Plot ROC curve
    plt.figure(figsize=(8,8))
    plt.plot(fpr, tpr, label='ROC curve - valid(AUC = %0.2f)' % roc_auc, color='r')
    plt.plot(fpr_1, tpr_1, label='ROC curve - train (AUC = %0.2f)' % roc_auc_1, color='b')
    plt.plot(fpr_2, tpr_2, label='ROC curve - test (AUC = %0.2f)' % roc_auc_2, color='g')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for lead score model')
    plt.legend(loc="lower right")
    plt.show()
 
       
