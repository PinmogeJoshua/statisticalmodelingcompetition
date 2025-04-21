import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 导入自定义模块
from data_loader import load_sentiment_data, preprocess_text, get_feature_importance_data
from satisfaction_model import build_satisfaction_model
from feature_analysis import analyze_feature_factors, identify_key_factors
from results_discussion import evaluate_and_discuss
from deep_learning_model import build_deep_learning_model, compare_with_traditional_models
# 删除BERT相关的导入
import traceback

def main():
    print("第六章：用户满意度模型与特征因素分析")
    
    # 创建结果目录
    os.makedirs('results', exist_ok=True)
    
    # 第一步：加载和预处理数据
    print("\n1. 数据加载与预处理...")
    df = load_sentiment_data()
    df = preprocess_text(df)
    print(f"- 加载了共{len(df)}条评论")
    print(f"- 产品数量: {df['product'].nunique()}")
    print(f"- 正面评论: {len(df[df['sentiment'] == 1])}")
    print(f"- 负面评论: {len(df[df['sentiment'] == -1])}")
    
    # 应用增强特征工程
    print("\n1.1 应用增强特征工程...")
    from data_loader import enhanced_feature_engineering
    df = enhanced_feature_engineering(df)
    print(f"- 特征工程后的特征数量: {len(df.columns)}")
    
    # 获取特征因素数据
    feature_data = get_feature_importance_data()
    
    # 第二步：可视化各产品的正负面评论比例
    print("\n2. 可视化产品评论情感分布...")
    product_sentiment = pd.crosstab(df['product'], df['sentiment'])
    product_sentiment.columns = ['负面', '正面']
    
    plt.figure(figsize=(14, 8))
    product_sentiment.plot(kind='bar', color=['salmon', 'skyblue'], figsize=(14, 8))
    plt.title('各产品正面与负面评论数量对比')
    plt.xlabel('产品')
    plt.ylabel('评论数量')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('results/product_sentiment_distribution.png', dpi=300)
    
    # 计算正面评论占比
    product_sentiment['正面占比'] = product_sentiment['正面'] / (product_sentiment['正面'] + product_sentiment['负面']) * 100
    print("各产品正面评论占比:")
    print(product_sentiment['正面占比'].sort_values(ascending=False).round(1))
    
    # 保存产品情感分析结果
    product_sentiment.to_csv('results/product_sentiment_analysis.csv')
    
    # 第三步：构建用户满意度模型
    print("\n3. 构建用户满意度模型...")
    model_results = build_satisfaction_model(df)
    
    # 第四步：特征因素重要性分析
    print("\n4. 特征因素重要性分析...")
    feature_analysis_results = analyze_feature_factors(df, model_results, feature_data)
    
    # 识别关键特征因素
    print("\n5. 识别关键特征因素...")
    key_factors = identify_key_factors(feature_analysis_results)
    feature_analysis_results['key_factors'] = key_factors
    
    # 第六步：结果评估与讨论
    print("\n6. 结果评估与讨论...")
    discussion_results = evaluate_and_discuss(df, model_results, feature_analysis_results)

    # 删除BERT分析部分
    
    # 第七步：保存原始数据与分析结果
    print("\n7. 保存分析结果...")
    
    # 保存原始预处理数据
    df.to_csv('results/processed_data.csv', index=False)
    
    # 生成简要分析摘要
    with open('results/analysis_summary.txt', 'w', encoding='utf-8') as f:
        f.write("第六章：用户满意度模型与特征因素分析 - 结果摘要\n")
        f.write("=======================================\n\n")
        
        f.write("1. 数据统计\n")
        f.write(f"   - 总评论数: {len(df)}\n")
        f.write(f"   - 产品数量: {df['product'].nunique()}\n")
        f.write(f"   - 正面评论: {len(df[df['sentiment'] == 1])}\n")
        f.write(f"   - 负面评论: {len(df[df['sentiment'] == -1])}\n\n")
        
        f.write("2. 产品情感分析\n")
        for product, pos_ratio in product_sentiment['正面占比'].sort_values(ascending=False).items():
            f.write(f"   - {product}: {pos_ratio:.1f}% 正面评论\n")
        f.write("\n")
        
        f.write("3. 最佳用户满意度模型\n")
        f.write(f"   - 模型: {model_results['best_model_name']}\n")
        f.write(f"   - 准确率: {model_results['model_results'][model_results['best_model_name']]['accuracy']:.4f}\n\n")
        
        f.write("4. 特征因素重要性排名\n")
        for i, row in key_factors.iterrows():
            f.write(f"   - {row['特征因素']}: {row['综合得分']:.4f}\n")
        f.write("\n")
        
        f.write("5. 产品综合评分排名\n")
        for product, score in feature_analysis_results['product_rankings'].items():
            f.write(f"   - {product}: {score:.4f}\n")
        f.write("\n")
        
        f.write("6. 产品改进建议摘要\n")
        for product, data in discussion_results['product_improvements'].items():
            f.write(f"   - {product}:\n")
            f.write(f"     优势: {', '.join(data['strengths'])}\n")
            f.write(f"     劣势: {', '.join(data['weaknesses'])}\n")
            for rec in data['recommendations']:
                f.write(f"     建议: {rec}\n")
            f.write("\n")
            
        print("\n7. 使用深度学习模型进行分析...")
        dl_results = build_deep_learning_model(df)

        # 比较深度学习模型与传统模型性能
        print("\n8. 比较深度学习与传统模型性能...")
        model_comparison = compare_with_traditional_models(dl_results, model_results)

        # 将比较结果添加到分析摘要
        with open('results/analysis_summary.txt', 'a', encoding='utf-8') as f:
            f.write("\n7. 深度学习模型分析\n")
            f.write(f"   - 模型准确率: {dl_results['accuracy']:.4f}\n")
            f.write("   - 模型性能比较：\n")
            for i, row in model_comparison.iterrows():
                f.write(f"     {row['模型']}: {row['准确率']:.4f}\n")
            f.write("\n")
    
    print("\n所有结果已统一保存至results文件夹")
    print("第六章分析完成！")

if __name__ == "__main__":
    main()
