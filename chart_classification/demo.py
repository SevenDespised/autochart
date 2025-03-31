"""
ChartClassifier 功能演示脚本

此脚本展示了 ChartClassifier 类的主要功能，包括：
1. 模型训练
2. 模型预测与评估
3. 参数搜索
4. 单条数据预测
"""

import os
import argparse
from chart_classifier import ChartClassifier


def setup_directories():
    """确保必要的目录存在"""
    dirs = [
        'chart_classifi_models',
        'chart_classification/output',
        'chart_classification/log'
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("目录检查完成")


def demo_train():
    """模型训练演示"""
    print("\n=== 模型训练演示 ===")
    classifier = ChartClassifier()
    model = classifier.train(save_model=True)
    print(f"模型训练完成，特征重要性前5项：")
    importance = model.feature_importance()
    feature_names = model.feature_name()
    if feature_names and importance is not None:
        importance_dict = dict(zip(feature_names, importance))
        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        for feat, imp in sorted_importance[:5]:
            print(f"  {feat}: {imp}")
    return classifier, model


def demo_predict(classifier=None, model=None):
    """模型预测演示"""
    print("\n=== 模型预测演示 ===")
    if classifier is None:
        classifier = ChartClassifier()
    if model is None:
        print("加载已保存的模型...")
        
    y_pred, metrics = classifier.predict(model=model)
    print(f"预测完成，评估指标：")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    return y_pred, metrics


def demo_parameter_search():
    """参数搜索演示"""
    print("\n=== 参数搜索演示 ===")
    classifier = ChartClassifier()
    
    # 定义多组参数进行搜索
    search_configs = [
        {'param_key': 'num_trees', 'start': 10, 'end': 20, 'step': 10},
        {'param_key': 'min_child_samples', 'start': 10, 'end': 50, 'step': 10}
    ]
    
    # 仅搜索一个参数节省时间
    search_config = search_configs[0]
    print(f"搜索参数: {search_config['param_key']} 从 {search_config['start']} 到 {search_config['end']}，步长 {search_config['step']}")
    
    results = classifier.parameter_search(**search_config)
    
    print(f"参数搜索完成，最佳值:")
    best_param = max(results.items(), key=lambda x: x[1].get('acc', 0))
    print(f"  {search_config['param_key']} = {best_param[0]}, 准确率: {best_param[1].get('acc', 0):.4f}")
    
    return results


def demo_predict_single():
    """单条数据预测演示"""
    print("\n=== 单条数据预测演示 ===")
    classifier = ChartClassifier()
    
    # 确保输出目录存在
    os.makedirs(classifier.paths['output'], exist_ok=True)
    
    # 加载图表类型映射
    mapping = classifier.load_chart_type_mapping()
    print(f"加载了 {len(mapping)} 种图表类型映射")
    
    # 预测单条数据
    print("开始预测单条数据...")
    results = classifier.predict_single(save_result=True)
    
    # 显示部分结果
    print("\n预测结果示例（前5行）:")
    if len(results) > 0:
        print(results.head(min(5, len(results))))
    else:
        print("没有预测结果")
    
    return results


def demo_all():
    """运行所有演示"""
    print("=== ChartClassifier 全功能演示 ===")
    setup_directories()
    
    # 训练模型
    classifier, model = demo_train()
    
    # 使用训练好的模型进行预测
    demo_predict(classifier, model)
    
    # 参数搜索
    demo_parameter_search()
    
    # 单条数据预测
    demo_predict_single()
    
    print("\n演示完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ChartClassifier 功能演示')
    parser.add_argument('--mode', type=str, default='all', 
                        choices=['all', 'train', 'predict', 'search', 'single'],
                        help='演示模式')
    args = parser.parse_args()
    
    args.mode = 'train'
    # 根据参数选择演示模式
    if args.mode == 'all':
        demo_all()
    elif args.mode == 'train':
        setup_directories()
        demo_train()
    elif args.mode == 'predict':
        setup_directories()
        demo_predict()
    elif args.mode == 'search':
        setup_directories()
        demo_parameter_search()
    elif args.mode == 'single':
        setup_directories()
        demo_predict_single()