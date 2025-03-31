# coding: utf-8

# In[1]:


import traceback
import pandas as pd

import os
from time import time

from features.single_field_features import extract_single_field_features
from data_loader.data_loader_single import load_data_single

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

def extract_features(data, data_id = "test:1", description=None):
    res = []
    fid = data_id
    print(f"[Data ID: {fid}]: {description}")
    num_fields = len(data)
    try:
        extraction_results = extract_features_from_fields(data.items(), fid=fid, num_fields=num_fields)
        res.extend(extraction_results)
    except Exception as e:
        print('Uncaught exception: {}:{}'.format(type(e), e))
        traceback.print_tb(e.__traceback__)
    return pd.DataFrame(res)

def write_batch_results(batch_results, features_dir_name, write_header=False):
    batch_field_level_features_dfs = []


    batch_field_level_features_dfs.append(batch_results)

    concatenated_results = {
        'field_level_features_df': pd.concat(batch_field_level_features_dfs, ignore_index=True) if batch_field_level_features_dfs else pd.DataFrame(),
    }
    
    for (k, v) in concatenated_results.items():
        output_file_name = os.path.join(features_dir_name, "single_" + f"{k}.csv")
        v.to_csv(output_file_name, mode='w', index=False, header=write_header)

data = load_data_single(data_file_name = base_dir + f"tmp_output/data.csv")
if not os.path.exists(os.path.join(base_dir, 'features')):
    os.mkdir(os.path.join(base_dir, 'features'))

features_dir_name = os.path.join(base_dir, 'features')

if not os.path.exists(features_dir_name):
    os.mkdir(features_dir_name)

    
start_time = time()
res = extract_features(data)

print('Total time: {:.2f}s'.format(time() - start_time))
write_batch_results(res, features_dir_name = features_dir_name, write_header = True)

# In[7]:



df_f = pd.read_csv(base_dir + f"features/single_field_level_features_df.csv")
df_f_dedup = df_f.drop_duplicates('field_id')
df_f_dedup_clean = df_f_dedup[df_f_dedup['exists']!='exists'].copy()
 
with open("feature_extraction/preserve_id_v4.pkl", 'rb') as f:
    _ = pickle.load(f)
with open("feature_extraction/feature_list_float_bool.pkl", 'rb') as f:
    feature_list = pickle.load(f)

df_f_dedup_clean.loc[:, 'trace_type'] = 'unknown'  # 默认值for trace_type
df_f_dedup_clean.loc[:, 'is_x_or_y'] = 'x'     # 默认值for is_x_or_y

def cut_off(x):
    if x >= quantile_1 and x <= quantile_3:
        return x
    elif x < quantile_1:
        return quantile_1
    else:
        return quantile_3

with open("feature_extraction/nl2chart_dict_cut_off.pkl", 'rb') as f:
    dict_cut_off = pickle.load(f)

for i in feature_list['float']:
    df_f_dedup_clean[i] = np.array(df_f_dedup_clean[i].astype('float32'))
    quantile_1, quantile_3 = dict_cut_off[i]
    df_f_dedup_clean[i] = df_f_dedup_clean[i].apply(cut_off)
df_f_dedup_clean.loc[:, feature_list['bool']] = df_f_dedup_clean[feature_list['bool']].fillna(False)

df_f_dedup_clean.to_csv((base_dir + f"features/single_data_feature.csv"), index=False)