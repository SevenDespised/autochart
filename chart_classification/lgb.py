import trace
import os
import logging
import logging.handlers

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder 
from sklearn.metrics import accuracy_score
from lightgbm import log_evaluation

KEY = 'learning_rate'
START = 100
END = 1000
STEP = 100

paths = {'train': 'features/nl2chart_feature_train.csv',
        'test': 'features/nl2chart_feature_test.csv',
        #'train': 'features/feature_train.csv',
        #'test': 'features/feature_test.csv',
        'model_path': 'chart_classifi_models/',
        #'default_model': 'chart_classifi_models/model.txt',
        'default_model': 'chart_classifi_models/model_nl2chart.txt',
        #'log': 'src/log/train.log'}
        'log': 'chart_classfication/log/batch_test_param.log'}
#param = {'num_leaves':31, 'num_trees':100 , 'objective':'multiclass', 'num_class': 6, 'metric':'auc_mu'}
param = {
    'task': 'train',
    # 'boosting_type': 'rf',  
    'boosting_type': 'gbdt',  
    'objective': 'multiclass',
    'metric': 'auc_mu',
    'num_class': 7, # VizMl data: 6  nl2chart data: 7
    'max_bin': 255, 
    'learning_rate': 0.1,
    'num_leaves': 3,
    'num_trees': 100,
    'max_depth': 2,
    'min_child_samples': 40,
    'feature_fraction': 0.8,
    'bagging_freq': 5, 
    'bagging_fraction': 0.8,
    #'min_data_in_leaf': 5,
    'min_sum_hessian_in_leaf': 3.0,
}
callbacks = [log_evaluation(period=100)]
#callbacks = [log_evaluation(period=100), early_stopping(stopping_rounds=30)]

logger = logging.getLogger('train')
logger.setLevel(level=logging.DEBUG)

formatter = logging.Formatter('%(asctime)s -- %(levelname)s: %(message)s')
file_handler = logging.handlers.TimedRotatingFileHandler(paths['log'], when = 'H')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

def ParamTest(parameter, key, start, end, step, func, **kwargs):
    cnt = 1
    paths['log'] = 'src/log/batch_test_param.log'
    for para in np.arange(start, end, step):
        parameter[key] = para
        _, m = func(**kwargs)
        
        logger.info('#######\n[epoch %s]: %s = %s,\n%s\n', cnt, key, para, m)
        cnt += 1
    

def LoadDataset(path, mode = 'test'):
    if mode not in ['test', 'train', 'val']:
        raise "Please pass string 'test', 'train' or 'val' as the parameter, but actually it doesnt work."
    
    data = pd.read_csv(path)

    truth = None
    if mode != 'predict':
        def pop_with_default(dataframe, column_name, default = None):
            if column_name in dataframe.columns:
                return dataframe.pop(column_name)
            return default
        pop_with_default(data, 'field_id')
        pop_with_default(data, 'fid')
        pop_with_default(data, 'is_xsrc')
        pop_with_default(data, 'is_ysrc')
        pop_with_default(data, 'trace_type_n')
        is_x_or_y = pop_with_default(data, 'is_x_or_y')
        trace_type = pop_with_default(data, 'trace_type')

        truth = is_x_or_y.to_frame()
        truth.insert(loc=truth.shape[1], column='trace_type', value=trace_type, allow_duplicates=False)
        #data.insert(loc=data.shape[1], column='is_xsrc', value=is_xsrc, allow_duplicates=False)
        #data.insert(loc=data.shape[1], column='is_ysrc', value=is_ysrc, allow_duplicates=False)
        #data.insert(loc=data.shape[1], column='trace_type', value=trace_type, allow_duplicates=False)
        
    return data, truth

def TrainModel(data_path, model_path, parameter):
    X, Y = LoadDataset(data_path)
    le = LabelEncoder()
    Y['is_x_or_y'] = le.fit_transform(Y['is_x_or_y'].astype('str'))
    Y['trace_type'] = le.fit_transform(Y['trace_type'].astype('str'))  
    #object_label = [label for label, ser in X.items() if ser.dtype == 'object']
    #X[object_label] = X[object_label].astype('category')
    #print(X.shape, Y.shape)
    
    x_train, x_test, y_train, y_test = train_test_split(X, Y['trace_type'], test_size=0.2, random_state=100)
    train_data = lgb.Dataset(data = x_train, label = y_train)
    test_data = lgb.Dataset(data = x_test, label = y_test)
    
    bst = lgb.train(parameter, 
                    train_data,
                    valid_sets=[test_data], 
                    callbacks = callbacks)

    bst.save_model(model_path)
    #print(bst.feature_importance())
   
def LoadModel(model_path):
    return lgb.Booster(model_file=model_path)

def Predict(model_path, paths, only_metrics = False, metrics = ['acc'], retrain = False, parameter = param):
    test_data_path = paths['test']
    train_data_path = paths['train']
    if retrain:
        model_path = os.path.join(paths['model_path'], GetModelName(param))
        TrainModel(train_data_path, model_path, parameter)
    else:
        model_path = paths['default_model']
    bst = LoadModel(model_path) 

    le = LabelEncoder()
    
    test_data, truth_data = LoadDataset(test_data_path)
    y_true = le.fit_transform(truth_data['trace_type'].astype('str'))
    #print('[PredictInfo]: Input test data shape is {}'.format(test_data.shape))
    
    y_pred = bst.predict(test_data, num_iteration=bst.best_iteration)
    y_pred_deco = le.inverse_transform(np.argmax(y_pred, axis = 1))
    metrics = GetMetrics(y_pred, y_true, metrics)
    #y_pred = np.around(y_pred,0).astype(int)
    logger.debug('\n-- Metrics:\n%s\n-- Parameters:\n%s\n-- Model:\n%s\n', metrics, parameter, model_path)
    print(bst.feature_importance())
    return metrics if only_metrics else (y_pred_deco, metrics)

def GetMetrics(y_pred, y_true, metrics):
    result = {}
    n = len(y_pred)
    cnt1 = 0
    cnt2 = 0
    mr = 0
    if 'acc' in metrics:
        result['acc'] = accuracy_score(y_true, np.argmax(y_pred, axis = 1)) 
    
    if 'hit@2'in metrics:
        # max_rank = len(y_pred[0]) - 1
        cnt = 0
        for i in range(n):
            idxes = np.argsort(-y_pred[i])
            for rank, idx in enumerate(idxes):
                y_pred[i][idx] = rank
            rank_p = y_pred[i][y_true[i]]
            if  rank_p <= 0:
                cnt1 += 1
            if  rank_p <= 1:
                cnt2 += 1
            mr += rank_p
        result['hit@1'] = float(cnt1) / float(n)
        result['hit@2'] = float(cnt2) / float(n)
        
    if  'MR' in metrics:
        # the range of rank in code is [0, n - 1] but show [1, n]
        result['MR'] = float(mr) / float(n) + 1
    return result
        
   
def GetTimeStamp():
    from datetime import datetime
    return datetime.now().strftime("%m%d%H%M%S")

def GetParamString(param):
    try:
        s = 'leave_{}_tree_{}_'.format(param['num_leaves'], param['num_trees'])
    except:
        s = ''
    return s

def GetModelName(param):
    return "model_" + GetParamString(param) + GetTimeStamp() + ".txt"
    
if __name__ == "__main__":
    model_path = os.path.join(paths['model_path'], GetModelName(param))
    input_param = {'model_path': model_path, 
                   'paths': paths, 
                   'metrics': ['acc', 'hit@2', 'MR'],
                   'only_metrics': False,
                   'retrain': True}
    #ParamTest(parameter = param, key = KEY, start = START, end = END, step = STEP, func = Predict, **input_param)
    y, m = Predict(model_path, paths, metrics = ['acc', 'hit@2', 'MR'], only_metrics = False, retrain = True)
    print("Metrics:" + str(m))