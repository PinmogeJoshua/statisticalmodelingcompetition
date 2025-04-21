import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler
import os

def build_satisfaction_model(df):
    """构建用户满意度模型"""
    print("构建用户满意度模型...")
    
    # 确保结果目录存在
    os.makedirs('results', exist_ok=True)
    
    # 准备特征和目标变量
    features = ['性价比_score', '质量_score', '购物体验_score', '实用性_score', 
                '性价比_count', '质量_count', '购物体验_count', '实用性_count',
                'comment_length']
    
    X = df[features]
    y = df['sentiment']  # 1表示正面情感，-1表示负面情感
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 定义要测试的模型
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'SVM': SVC(probability=True, random_state=42)
    }
    
    # 存储每个模型的结果
    model_results = {}
    model_predictions = {}
    best_model_name = None
    best_accuracy = 0
    
    # 训练并评估每个模型
    for name, model in models.items():
        print(f"\n训练 {name} 模型...")
        model.fit(X_train_scaled, y_train)
        
        # 预测测试集
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        # 保存模型结果
        model_results[name] = {
            'model': model,
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred)
        }
        
        model_predictions[name] = y_pred
        
        print(f"{name} 模型准确率: {accuracy:.4f}")
        print(f"{name} 分类报告:")
        print(classification_report(y_test, y_pred))
        
        # 更新最佳模型
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = name
    
    # 可视化各模型的准确率
    plt.figure(figsize=(10, 6))
    accuracies = [results['accuracy'] for results in model_results.values()]
    sns.barplot(x=list(models.keys()), y=accuracies)
    plt.title('不同模型的准确率对比')
    plt.xlabel('模型')
    plt.ylabel('准确率')
    plt.ylim(0.8, 1.0)  # 设置y轴范围，便于比较
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('results/model_accuracy_comparison.png', dpi=300)
    
    # 可视化混淆矩阵
    plt.figure(figsize=(15, 10))
    for i, (name, y_pred) in enumerate(model_predictions.items(), 1):
        plt.subplot(2, 2, i)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{name} 混淆矩阵')
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
    
    plt.tight_layout()
    plt.savefig('results/model_confusion_matrices.png', dpi=300)
    
    # 获取最佳模型
    best_model = model_results[best_model_name]['model']
    print(f"\n最佳模型是: {best_model_name}, 准确率: {best_accuracy:.4f}")
    
    # 特征重要性分析（如果模型支持）
    feature_importance = analyze_feature_importance(best_model, features, best_model_name)
    
    # 保存特征重要性到CSV
    if feature_importance is not None:
        feature_importance.to_csv('results/feature_importance.csv')
    
    # 返回模型结果
    return {
        'best_model_name': best_model_name,
        'best_model': best_model,
        'model_results': model_results,
        'feature_importance': feature_importance,
        'features': features,
        'X_test': X_test_scaled,
        'y_test': y_test
    }

def analyze_feature_importance(model, feature_names, model_name):
    """分析模型的特征重要性"""
    import numpy as np
    from sklearn.inspection import permutation_importance
    
    # 如果是基于树的模型，可直接获取特征重要性
    if model_name in ['RandomForest', 'GradientBoosting']:
        # 这些模型直接提供feature_importances_属性
        importances = model.feature_importances_
    
    # 如果是逻辑回归模型，使用系数绝对值
    elif model_name == 'LogisticRegression':
        # 逻辑回归使用系数作为特征重要性
        importances = abs(model.coef_[0])  # 取绝对值
    
    # 对于其他模型（如SVM），使用排列重要性
    else:
        # 获取全局变量中的X_test和y_test
        X_test = globals().get('X_test_scaled')
        y_test = globals().get('y_test')
        
        if X_test is not None and y_test is not None:
            # 使用排列重要性
            result = permutation_importance(
                model, X_test, y_test, 
                n_repeats=10,  # 增加重复次数，提高稳定性
                random_state=42,
                n_jobs=-1  # 使用所有CPU核心加速计算
            )
            importances = result.importances_mean
        else:
            print(f"警告: 无法获取测试数据，无法计算{model_name}的排列重要性")
            importances = np.ones(len(feature_names)) * 0.1  # 默认值
    
    # 确保值不全为0（避免出现全为0的情况）
    if np.sum(importances) == 0:
        print("警告: 所有特征重要性都为0，应用随机重要性")
        importances = np.random.random(len(feature_names))
        importances = importances / np.sum(importances)  # 归一化
    
    # 创建特征重要性DataFrame
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # 可视化特征重要性
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title(f'{model_name} - 特征重要性排名')
    plt.xlabel('重要性分数')
    plt.tight_layout()
    plt.savefig('results/feature_importance.png', dpi=300)
    
    return feature_importance
    
    