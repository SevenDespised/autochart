# coding: utf-8

# In[1]:


import traceback
import pandas as pd

import os
from time import time

from features.single_field_features import extract_single_field_features
from data_loader.data_loader_nl2chart import load_data_nl2chart, data_format_trans

import pickle
import numpy as np
from sklearn.model_selection import train_test_split


MAX_FIELDS = 25
total_charts = 0
charts_without_data = 0
chart_loading_errors = 0
feature_extraction_errors = 0
charts_exceeding_max_fields = 0
CHUNK_SIZE = 1000

base_dir = ""

# In[2]:


def extract_features_from_fields(fields, fid=None, num_fields=2):

#     feature_names_by_type = {
#         'basic': ['fid'],
#         'single_field': [],
#         'field_outcomes': []
#     }

    single_field_features, _ = extract_single_field_features(
            fields, fid, MAX_FIELDS=MAX_FIELDS, num_fields=num_fields)

    df_field_level_features = []
    for f in single_field_features:
        if f['exists']:
            df_field_level_features.append(f)

    return df_field_level_features

def extract_features(data, chunk_size):
    res = []
    start_time = time()
    for i, id in enumerate(data):
        fid = id
        table_data = data[id]
        if i % chunk_size == 0:
            print('[Chunk %s][%s] %.1f: %s' %
                  (i % chunk_size + 1, i, time() - start_time, 'dataid:{}, database id:{}'.format(fid, table_data["db_id"])))
        vis_obj = table_data["vis_obj"]

        data_converted = data_format_trans(vis_obj)
        num_fields = len(data_converted)
        try:
            extraction_results = extract_features_from_fields(data_converted.items(), fid=fid, num_fields=num_fields)
            res.extend(extraction_results)
        except Exception as e:
            print('Uncaught exception: {}:{}'.format(type(e), e))
            traceback.print_tb(e.__traceback__)
            continue
    return pd.DataFrame(res)

def write_batch_results(batch_results, features_dir_name, write_header=False):
    batch_field_level_features_dfs = []


    batch_field_level_features_dfs.append(batch_results)

    concatenated_results = {
        'field_level_features_df': pd.concat(batch_field_level_features_dfs, ignore_index=True) if batch_field_level_features_dfs else pd.DataFrame(),
    }
    
    for (k, v) in concatenated_results.items():
        output_file_name = os.path.join(features_dir_name, "nl2chart_" + f"{k}.csv")
        v.to_csv(output_file_name, mode='a', index=False, header=write_header)

data = load_data_nl2chart(data_file_name = base_dir + f"feature_extraction/tmp_output/data.csv")
if not os.path.exists(os.path.join(base_dir, 'features')):
    os.mkdir(os.path.join(base_dir, 'features'))

features_dir_name = os.path.join(base_dir, 'features')

if not os.path.exists(features_dir_name):
    os.mkdir(features_dir_name)

    
start_time = time()
res = extract_features(data, 100)

print('Total time: {:.2f}s'.format(time() - start_time))
write_batch_results(res, features_dir_name = features_dir_name, write_header = True)

# In[7]:



df_f = pd.read_csv(base_dir + f"features/nl2chart_field_level_features_df.csv")
df_f_dedup = df_f.drop_duplicates('field_id')
df_f_dedup_clean = df_f_dedup[df_f_dedup['exists']!='exists']
 
with open("feature_extraction/preserve_id_v4.pkl", 'rb') as f:
    _ = pickle.load(f)
with open("feature_extraction/feature_list_float_bool.pkl", 'rb') as f:
    feature_list = pickle.load(f)


def get_chart_type(fid):
    return data[fid]['chart']
def get_is_x_or_y(filed_id):
    return filed_id.split(':')[1].split('#')[0]

df_f_dedup_clean['trace_type'] = df_f_dedup_clean['fid'].apply(get_chart_type)
df_f_dedup_clean['is_x_or_y'] = df_f_dedup_clean['field_id'].apply(get_is_x_or_y)
list_dataset_split = train_test_split(df_f_dedup_clean, train_size=0.7, test_size=0.3)
df_train, df_test    = list_dataset_split[0], list_dataset_split[1]


def cut_off(x):
    if x >= quantile_1 and x <= quantile_3:
        return x
    elif x < quantile_1:
        return quantile_1
    else:
        return quantile_3

dict_cut_off = {}

for i in feature_list['float']:

    df_train[i] = np.array(df_train[i].astype('float32'))
    quantile_1 = np.quantile(df_train[i][np.isfinite(df_train[i])], 0.05)
    quantile_3 = np.quantile(df_train[i][np.isfinite(df_train[i])], 0.95)

    dict_cut_off[i] = (quantile_1, quantile_3)

    df_train[i] = df_train[i].apply(cut_off)

df_train[feature_list['bool']] = df_train[feature_list['bool']].fillna(False)
df_test[feature_list['bool']] = df_test[feature_list['bool']].fillna(False)

for i in feature_list['float']:
    
    df_test[i] = np.array(df_test[i].astype('float32'))
    quantile_1, quantile_3 = dict_cut_off[i]
    df_test[i] = df_test[i].apply(cut_off)


df_train.to_csv((base_dir + f"features/nl2chart_feature_train.csv"), index=False)
df_test.to_csv((base_dir + f"features/nl2chart_feature_test.csv"), index=False)