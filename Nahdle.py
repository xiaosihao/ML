def append_dic(dic,key,value):
	'''
	To append value to a given key as a element of a list

	Inputs
	----------
	dic: destination dictionary
	key: destination key
	value: value that wanted to apped to the key
	
	Output
	----------
	Add the value to the given key 
	'''
	if key not in dic.keys():
		dic[key] = [value]
	else:
		l = dic[key]
		l.append(value)
		dic[key] = l


def nahdle(clf, data, log_path, data_path, splits=5, max_iter=10, mask_threshold=0.6):
    '''
    Spontaneously removing features with 0 importance with cross validation

    Inputs
    ----------
    clf: classification model with full hyperparameters
    data: training dataset in pandas dataframe format
    log_path: output log pathway
    data_path: output dataframe pathway
    splits: number of splits for cross validation (number of folds)
    max_iter: number of times to perform the removing process
    mask_threshold: minimum

    Outputs
    ----------
    The json format log file containing all parameteres and results and evaluation scores for each iteration performed
    A feature selected dataframe saved as tsv format after each iteration
    The final dataframe after rounds of selections saved as tsv format

    Returns
    ----------

    '''
    #required modules
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    from collections import Counter
    import time
    from statistics import mean
    import json
    import re
    import pandas as pd
    from xiao import roc_stat

    #initial time
    init_time = time.time()
    #parameters
    log_dic = {}
    cv = StratifiedKFold(n_splits=splits)
    mask_occurance = mask_threshold*splits
    iter_id = 1
    print(f'Total interations: {max_iter}')
    print('--------------------')
    #log
    log_dic['Name']='Spontaneously removing features with 0 importance with cross validation'
    log_dic['Number of iterations']=max_iter
    log_dic['Cross validation fold']=splits
    log_dic['Threshold for masking']=mask_threshold
    log_dic['Occurance for masking']=mask_occurance
    #start
    while iter_id <= max_iter:
        print(f'Iteration {iter_id}')
        temp_dic = {}
        temp_dic['iteration ID']=iter_id
        #parameteres
        rocs = []
        scores = []
        start_time = time.time()
        index = []
        #read from dataframe
        X = np.array(data.drop(['phenotype'],axis = 1))
        y = np.array(data['phenotype'])

        # perform cross validation. Recrod features with 0 importance, accuracy and roc score of each iteration.
        for train, test in cv.split(X, y):
            #train model
            clf.fit(X[train], y[train])

            # index of features with 0 importance
            index.extend([idx for idx, importance_ in enumerate(clf.feature_importances_) if importance_ == 0])
            #or
            #index.extend(np.where(clf.feature_importances_ == 0)[0])
            
            # evaluation
            _, _, auc = roc_stat(X[test], y[test], clf)
            rocs.append(auc)
            scores.append(clf.score(X[test], y[test]))
            print('**',end = '')

        # select the final masked features
        occurance = dict(Counter(index))
        temp_dic['Current number of features']=len(data.columns)
        temp_dic['Number of features with 0 importance occured in any fold']=len(occurance.keys())
        del_list = [key for key, count in zip(occurance.keys(),occurance.values()) if count >= mask_occurance]
        temp_dic[f'Number of features with 0 importance occured in at least {str(int(mask_occurance))} folds']=len(del_list)
        temp_dic['Removed feature names']= list(data.columns[del_list])

        # remove masked features from the data
        data.drop(data.columns[del_list],axis = 1, inplace = True)
        temp_dic['Number of features after removal']=len(data.columns)

        # save the evalutaion score
        temp_dic['Accuracies']= scores
        temp_dic['Mean accuracy']= '{:.4f}'.format(mean(scores))
        temp_dic['ROC_AUCs'] = rocs
        temp_dic['Mean ROC_AUC']='{:.4f}'.format(mean(rocs))

        # save the improved dataset
        data.to_csv(f'{data_path}/table_98_nahdle_{iter_id}.tsv',sep='\t')

        # save running time
        duration = time.time()-start_time
        temp_dic['Iteration running time'] = '{:.4f}s'.format(duration)
        print('*****')

        #save log
        append_dic(log_dic,'Info',temp_dic)
        iter_id += 1

    #save the final dataset
    data.to_csv(f'{data_path}/table_98_nahdle_final.tsv',sep='\t')
    #total running time
    log_dic['Program running time'] = '{:.4f}s'.format(init_time-time.time())
    #save log
    with open(f'{log_path}/log_nahdle.json','w') as f:
        f.write(json.dumps(log_dic))


from sklearn.ensemble import RandomForestClassifier
import pandas as pd

data = pd.read_csv('./table_98_50.txt',sep = '\t', index_col=0)
data_path = '.'
log_path = '.'
clf = RandomForestClassifier(n_jobs=-1)
nahdle(clf, data, data_path, log_path, max_iter=2, splits=5)
