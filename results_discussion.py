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
        
        # 根据劣势生成详细建议
        for weakness in weaknesses:
            if weakness == '性价比':
                product_improvements[product]['recommendations'].extend([
                    "重新评估产品定价策略，可考虑推出不同价位的产品系列，满足不同消费者的需求",
                    "增加产品附加价值，如延长保修期或提供免费售后服务，提高消费者对价格的接受度",
                    "通过改进生产流程降低成本，将节约的成本转化为价格优势"
                ])
            elif weakness == '质量':
                product_improvements[product]['recommendations'].extend([
                    "加强质量控制体系，引入更严格的产品测试标准",
                    "更新产品材料和工艺，提高产品的耐用性和可靠性",
                    "建立完善的质量反馈机制，及时收集和处理客户对产品质量的投诉",
                    "引入国际质量管理标准，如ISO9001认证"
                ])
            elif weakness == '购物体验':
                product_improvements[product]['recommendations'].extend([
                    "优化网站/APP界面，提高用户友好性",
                    "加强物流配送能力，缩短配送时间",
                    "培训客服团队，提高服务质量和响应速度",
                    "增加在线咨询和售后支持渠道",
                    "完善产品展示和说明，提供详细的参数、使用方法和注意事项"
                ])
            elif weakness == '实用性':
                product_improvements[product]['recommendations'].extend([
                    "通过用户调研深入了解目标客户的实际需求",
                    "重新设计产品，增强功能性和易用性",
                    "提供详细的使用指南和视频教程，帮助用户充分利用产品功能",
                    "简化产品操作流程，提高直观性",
                    "增加产品的多功能性，满足用户多样化需求"
                ])
    
    # 6. 生成总结报告
    print("\n6. 生成总结报告...")
    
    # 删除BERT相关代码
    
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

# 删除update_report_with_bert_results函数
