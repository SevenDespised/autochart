"""模型配置文件"""

import os
import json
from typing import Dict, Any

def get_default_config() -> Dict[str, Any]:
    """获取默认配置"""
    return {
        "paths": {
            'train': 'features/py12nl2chart_feature_train.csv',
            'test': 'features/py12nl2chart_feature_test.csv',
            'predict': 'features/single_data_feature.csv',
            'model_path': 'chart_classifi_models/',
            'default_model': 'chart_classifi_models/model_nl2chart.txt',
            'output': 'chart_classification/output/',
            'log': 'chart_classification/log/batch_test_param.log'
        },
        "model_params": {
            "task": "train",
            "boosting_type": "gbdt", 
            "objective": "multiclass",
            "metric": "auc_mu",
            "num_class": 7,
            "max_bin": 255,
            "learning_rate": 0.1,
            "num_leaves": 3, 
            "num_trees": 100,
            "max_depth": 2,
            "min_child_samples": 40,
            "feature_fraction": 0.8,
            "bagging_freq": 5,
            "bagging_fraction": 0.8,
            "min_sum_hessian_in_leaf": 3.0
        },
        "training": {
            "test_size": 0.2,
            "random_state": 100,
            "log_period": 100
        }
    }

def load_config(config_path: str = None) -> Dict[str, Any]:
    """加载配置
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    config = get_default_config()
    
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            user_config = json.load(f)
            
        # 递归更新配置
        def update_dict(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = update_dict(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
            
        config = update_dict(config, user_config)
        
    return config