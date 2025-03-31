import pandas as pd
def load_data_single(data_file_name):
    print('Loading nl2chart data from %s' % data_file_name)
    return data_format_trans(pd.read_csv(data_file_name, encoding="utf-8", low_memory=False))

def data_format_trans(df):
    # 将df转换为字典格式
    res = {}
    for idx, col in enumerate(df.columns):
        res[col] = {
            "uid": col,
            "order": idx,
            "data": df[col].tolist()
        }
    return res