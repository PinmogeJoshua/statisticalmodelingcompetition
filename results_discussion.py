import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize

def evaluate_and_discuss(df, model_results, feature_analysis_results):
    """评估模型性能并讨论分析结果"""
    print("评估模型性能与结果讨论...")
    
    # 确保结果目录存在
    os.makedirs('results', exist_ok=True)
    
    # 1. 模型性能评估
    print("\n1. 模型性能评估...")
    
    # 获取最佳模型信息
    best_model_name = model_results['best_model_name']
    best_model = model_results['best_model']
    
    # 提取所有模型的准确率
    model_accuracies = {name: results['accuracy'] 
                        for name, results in model_results['model_results'].items()}
    
    # 比较各模型性能
    models_comparison = pd.DataFrame({
        '模型': model_accuracies.keys(),
        '准确率': model_accuracies.values()
    }).sort_values('准确率', ascending=False)
    
    print("模型性能比较：")
    print(models_comparison)
    
    # 2. ROC曲线分析（如果模型支持predict_proba）
    print("\n2. ROC曲线分析...")
    
    plt.figure(figsize=(10, 8))
    
    # 为每个支持概率预测的模型绘制ROC曲线
    for name, model_info in model_results['model_results'].items():
        model = model_info['model']
        if hasattr(model, 'predict_proba'):
            try:
                y_test = model_results['y_test']
                y_score = model.predict_proba(model_results['X_test'])
                
                # 将标签转换为二值形式（对二分类任务）
                y_test_bin = label_binarize(y_test, classes=[-1, 1])
                
                # 使用正类的概率
                y_score = y_score[:, 1] if y_score.shape[1] > 1 else y_score
                
                # 计算ROC曲线
                fpr, tpr, _ = roc_curve(y_test_bin, y_score)
                roc_auc = auc(fpr, tpr)
                
                plt.plot(fpr, tpr, lw=2, 
                         label=f'{name} (AUC = {roc_auc:.3f})')
            except:
                print(f"无法为 {name} 模型绘制ROC曲线")
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正例率 (FPR)')
    plt.ylabel('真正例率 (TPR)')
    plt.title('各模型ROC曲线比较')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/model_roc_curves.png', dpi=300)
    
    # 3. 特征重要性讨论
    print("\n3. 特征重要性讨论...")
    
    # 如果有特征重要性结果
    if 'feature_importance' in model_results and model_results['feature_importance'] is not None:
        feature_imp = model_results['feature_importance']
        top_features = feature_imp.head(5)
        
        print("最重要的5个特征:")
        print(top_features)
    
    # 4. 特征因素分析结果讨论
    print("\n4. 特征因素分析结果讨论...")
    
    # 获取关键特征因素及其重要性
    key_factors = feature_analysis_results.get('key_factors', None)
    if key_factors is not None:
        print("关键特征因素排名:")
        print(key_factors)
    
    # 产品表现分析
    product_scores = feature_analysis_results['product_factor_scores']
    product_rankings = feature_analysis_results['product_rankings']
    
    print("\n产品综合排名:")
    print(product_rankings)
    
    # 5. 生成产品改进建议
    print("\n5. 生成产品改进建议...")
    
    # 为每个产品识别优势和劣势
    product_improvements = {}
    for product in product_scores.index:
        scores = product_scores.loc[product]
        strengths = scores.nlargest(2).index.tolist()
        weaknesses = scores.nsmallest(2).index.tolist()
        
        product_improvements[product] = {
            'strengths': strengths,
            'weaknesses': weaknesses,
            'recommendations': []
        }
        
        # 根据劣势生成建议
        for weakness in weaknesses:
            if weakness == '性价比':
                product_improvements[product]['recommendations'].append(
                    "提高性价比：重新评估产品定价或增加产品价值，提高消费者感知的价值")
            elif weakness == '质量':
                product_improvements[product]['recommendations'].append(
                    "提升产品质量：改进产品材料和生产工艺，提高产品耐用性和可靠性")
            elif weakness == '购物体验':
                product_improvements[product]['recommendations'].append(
                    "优化购物体验：改进物流配送速度，加强客服培训，提升服务质量")
            elif weakness == '实用性':
                product_improvements[product]['recommendations'].append(
                    "增强实用性：深入了解用户需求，改进产品设计，提高易用性和功能性")
    
    # 6. 生成总结报告
    print("\n6. 生成总结报告...")
    
    # 删除使用bert_results的部分，改为在update_report_with_bert_results函数中处理
    
    with open('results/analysis_report.md', 'w', encoding='utf-8') as f:
        f.write("# 用户满意度模型与特征因素分析报告\n\n")
        
        f.write("## 1. 模型性能评估\n\n")
        f.write("### 1.1 各模型准确率比较\n\n")
        f.write(models_comparison.to_markdown(index=False) + "\n\n")
        f.write(f"最佳模型是 **{best_model_name}**，准确率达到 **{model_accuracies[best_model_name]:.4f}**\n\n")
        
        f.write("## 2. 特征因素重要性分析\n\n")
        
        # 提及率分析
        mention_df = feature_analysis_results['mention_analysis']
        f.write("### 2.1 特征因素提及率\n\n")
        f.write(mention_df.to_markdown(index=False) + "\n\n")
        f.write(f"最常被提及的特征因素是**{mention_df.iloc[0]['特征因素']}**，提及率为**{mention_df.iloc[0]['提及率(%)']:.2f}%**\n\n")
        
        # 情感影响分析
        impact_df = feature_analysis_results['sentiment_impact']
        f.write("### 2.2 特征因素对情感的影响\n\n")
        f.write(impact_df.to_markdown(index=False) + "\n\n")
        f.write(f"对情感影响最大的特征因素是**{impact_df.iloc[0]['特征因素']}**，当评论提及该因素时，正面评论比例提高了**{impact_df.iloc[0]['差异']:.4f}**\n\n")
        
        # 关键特征因素
        if key_factors is not None:
            f.write("### 2.3 关键特征因素综合评估\n\n")
            f.write(key_factors.to_markdown(index=False) + "\n\n")
            f.write(f"综合各项指标，最重要的特征因素是**{key_factors.iloc[0]['特征因素']}**，综合得分为**{key_factors.iloc[0]['综合得分']:.4f}**\n\n")
        
        # 产品分析和改进建议
        f.write("## 3. 产品分析与改进建议\n\n")
        f.write("### 3.1 产品综合评分排名\n\n")
        product_rank_df = pd.DataFrame({
            '产品': product_rankings.index,
            '综合评分': product_rankings.values
        })
        f.write(product_rank_df.to_markdown(index=False) + "\n\n")
        
        f.write("### 3.2 产品特征因素得分详情\n\n")
        f.write(product_scores.to_markdown() + "\n\n")
        
        f.write("### 3.3 产品改进建议\n\n")
        for product, data in product_improvements.items():
            f.write(f"#### {product}\n\n")
            f.write(f"**优势特征**：{', '.join(data['strengths'])}\n\n")
            f.write(f"**劣势特征**：{', '.join(data['weaknesses'])}\n\n")
            f.write("**改进建议**：\n\n")
            for rec in data['recommendations']:
                f.write(f"- {rec}\n")
            f.write("\n")
        
        f.write("## 4. 结论\n\n")
        f.write("通过对用户评论的分析，我们构建了高准确率的用户满意度预测模型，并识别出影响用户满意度的关键特征因素。基于分析结果，我们为各产品提出了针对性的改进建议，以提升用户满意度。\n\n")
        f.write("研究表明，产品的质量和性价比是影响用户满意度的最重要因素，企业应重点关注这些方面的提升。同时，不同产品在各特征因素上的表现差异明显，需要针对各自的劣势方面进行有针对性的改进。\n\n")
        
    print("总结报告已生成：results/analysis_report.md")
    
    # 返回讨论结果
    return {
        'model_comparison': models_comparison,
        'product_improvements': product_improvements,
    }
    
def update_report_with_bert_results(df, bert_results):
    """将BERT分析结果更新到报告中"""
    if not bert_results:
        print("没有BERT分析结果可添加到报告")
        return
        
    print("将BERT分析结果添加到报告...")
    
    with open('results/analysis_report.md', 'a', encoding='utf-8') as f:
        f.write("\n## 5. BERT用户偏好分析结果\n\n")
        
        f.write("通过深度语义分析，我们将用户按偏好特征分为不同群体:\n\n")
        
        if 'cluster_preferences' in bert_results:
            # 添加用户群体偏好表
            f.write("### 5.1 用户群体偏好特征\n\n")
            f.write(bert_results['cluster_preferences'].to_markdown() + "\n\n")
            
            # 统计每个群体的产品分布
            if 'user_clusters' in bert_results:
                user_df = df.copy()
                user_df['cluster'] = bert_results['user_clusters']
                
                f.write("### 5.2 用户群体与产品关系\n\n")
                product_cluster_dist = pd.crosstab(
                    user_df['product'], 
                    user_df['cluster'], 
                    normalize='index'
                ) * 100
                
                f.write("下表展示了各产品用户在不同偏好群体中的分布百分比:\n\n")
                f.write(product_cluster_dist.to_markdown() + "\n\n")
                
                f.write("### 5.3 基于BERT分析的营销建议\n\n")
                
                for product in df['product'].unique():
                    f.write(f"#### {product}\n\n")
                    
                    if product in product_cluster_dist.index:
                        main_cluster = product_cluster_dist.loc[product].idxmax()
                        main_pct = product_cluster_dist.loc[product, main_cluster]
                        
                        f.write(f"- **主要用户群体**: 群体 {main_cluster} (占比: {main_pct:.1f}%)\n")
                        
                        if main_cluster in bert_results['cluster_preferences'].index:
                            prefs = bert_results['cluster_preferences'].loc[main_cluster]
                            top_features = prefs.nlargest(2).index.tolist()
                            
                            f.write(f"- **核心关注点**: {', '.join(top_features)}\n")
                            f.write(f"- **营销建议**: 突出产品的{top_features[0]}特性，强化{top_features[1]}方面的体验\n\n")
        
        f.write("详细的BERT用户偏好分析请参考 [BERT用户偏好分析报告](bert_preference_analysis.md)\n\n")
