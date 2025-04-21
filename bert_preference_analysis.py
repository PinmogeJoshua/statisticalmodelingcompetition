import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import os

# 导入本地模型加载器
from bert_model_loader import load_local_bert_model as load_bert_model

def extract_bert_embeddings(df, tokenizer, model, batch_size=16):
    """使用BERT提取评论的语义嵌入"""
    print("使用BERT提取评论语义嵌入...")
    
    # 确保评论列为文本
    comments = df['comment'].astype(str).tolist()
    
    # 存储评论嵌入
    embeddings = []
    
    # 批量处理，避免内存溢出
    for i in range(0, len(comments), batch_size):
        batch_comments = comments[i:i+batch_size]
        
        # 分词并转换为模型输入
        inputs = tokenizer(batch_comments, padding=True, truncation=True, max_length=128, return_tensors="pt")
        
        # 获取BERT输出
        with torch.no_grad():
            outputs = model(**inputs)
            
        # 使用[CLS]标记的输出作为整个句子的嵌入表示
        batch_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
        embeddings.extend(batch_embeddings)
        
        if (i + batch_size) % 100 == 0 or i + batch_size >= len(comments):
            print(f"处理进度: {i + batch_size}/{len(comments)}")
    
    # 将嵌入转换为numpy数组
    return np.array(embeddings)

def analyze_user_preferences(df, feature_data=None):
    """使用BERT进行用户偏好分析"""
    print("开始BERT用户偏好分析...")
    
    # 确保结果目录存在
    os.makedirs('results', exist_ok=True)
    
    try:
        # 加载本地BERT模型
        tokenizer, model = load_bert_model()
        if tokenizer is None or model is None:
            raise ValueError("BERT模型加载失败，无法进行分析")
        
        # 提取评论嵌入
        embeddings = extract_bert_embeddings(df, tokenizer, model)
        
        # 使用降维技术将高维BERT嵌入降到2维进行可视化
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
        
        # 使用PCA初步降维以加速t-SNE
        pca = PCA(n_components=min(50, embeddings.shape[1], embeddings.shape[0]))
        reduced_embeddings = pca.fit_transform(embeddings)
        
        # 使用t-SNE进一步降维到2维
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(df)/10))
        embeddings_2d = tsne.fit_transform(reduced_embeddings)
        
        # 添加2D嵌入到DataFrame
        df_with_embeddings = df.copy()
        df_with_embeddings['embedding_x'] = embeddings_2d[:, 0]
        df_with_embeddings['embedding_y'] = embeddings_2d[:, 1]
        
        # 可视化评论嵌入空间，按产品和情感分类
        plt.figure(figsize=(12, 10))
        
        # 为不同产品分配不同颜色
        products = df['product'].unique()
        product_colors = dict(zip(products, plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(products)))))
        
        for product in products:
            product_df = df_with_embeddings[df_with_embeddings['product'] == product]
            
            # 分别绘制正面和负面评论
            pos_reviews = product_df[product_df['sentiment'] == 1]
            neg_reviews = product_df[product_df['sentiment'] == -1]
            
            plt.scatter(pos_reviews['embedding_x'], pos_reviews['embedding_y'], 
                       color=product_colors[product], marker='o', alpha=0.6, 
                       label=f'{product} (正面)')
            
            plt.scatter(neg_reviews['embedding_x'], neg_reviews['embedding_y'], 
                       color=product_colors[product], marker='x', alpha=0.6)
        
        plt.title('BERT评论嵌入空间可视化')
        plt.xlabel('t-SNE维度1')
        plt.ylabel('t-SNE维度2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig('results/bert_comment_embeddings.png', dpi=300)
        
        # 聚类分析用户偏好
        print("\n使用聚类分析用户评论...")
        kmeans = KMeans(n_clusters=min(5, len(df)//100+1), random_state=42)
        df_with_embeddings['cluster'] = kmeans.fit_predict(embeddings)
        
        # 计算每个聚类的特征偏好得分
        feature_factors = ['性价比', '质量', '购物体验', '实用性']
        cluster_preferences = {}
        
        # 创建特征关键词列表，用于文本匹配
        feature_keywords = {
            '性价比': ['性价比', '价格', '便宜', '实惠', '划算', '物美价廉', '经济', '实在'],
            '质量': ['质量', '做工', '材质', '耐用', '可靠', '精细', '坚固', '牢靠', '好用'],
            '购物体验': ['服务', '物流', '送货', '包装', '客服', '态度', '速度', '体验', '快递'],
            '实用性': ['实用', '功能', '方便', '好用', '设计', '操作', '使用', '效果', '效率']
        }
        
        # 使用词频统计方法计算偏好得分
        for cluster in df_with_embeddings['cluster'].unique():
            cluster_df = df_with_embeddings[df_with_embeddings['cluster'] == cluster]
            cluster_comments = cluster_df['comment'].astype(str).tolist()
            cluster_preferences[cluster] = {}
            
            # 统计每个评论中各特征的提及情况
            for factor in feature_factors:
                keywords = feature_keywords[factor]
                factor_mentions = 0
                factor_sentiment = 0
                
                # 遍历该聚类的所有评论
                for i, comment in enumerate(cluster_comments):
                    # 计算特征关键词的出现次数
                    mention_count = sum(1 for keyword in keywords if keyword in comment)
                    sentiment = cluster_df.iloc[i]['sentiment'] if mention_count > 0 else 0
                    
                    factor_mentions += mention_count
                    factor_sentiment += sentiment * mention_count
                
                # 计算特征偏好得分 = 提及次数 * 情感系数
                if factor_mentions > 0:
                    # 归一化特征得分: 提及频率 * 情感倾向
                    score = (factor_mentions / len(cluster_comments)) * (factor_sentiment / factor_mentions + 1)
                else:
                    score = 0.01  # 避免完全为0
                    
                cluster_preferences[cluster][factor] = score
        
        # 将聚类偏好转换为DataFrame并可视化
        cluster_pref_df = pd.DataFrame(cluster_preferences).T
        
        # 创建热图
        plt.figure(figsize=(10, 8))
        sns.heatmap(cluster_pref_df, annot=True, cmap='YlGnBu', fmt='.2f')
        plt.title('用户聚类的特征偏好分析')
        plt.tight_layout()
        plt.savefig('results/bert_cluster_preferences.png', dpi=300)
        
        # 为每个聚类创建雷达图
        plt.figure(figsize=(15, 12))
        
        # 设置雷达图参数
        angles = np.linspace(0, 2*np.pi, len(feature_factors), endpoint=False).tolist()
        angles += angles[:1]  # 闭合雷达图
        
        # 确保数据被归一化以适合雷达图
        scaler = MinMaxScaler()
        cluster_pref_normalized = pd.DataFrame(
            scaler.fit_transform(cluster_pref_df),
            columns=cluster_pref_df.columns,
            index=cluster_pref_df.index
        )
        
        # 为每个聚类绘制雷达图
        n_clusters = len(cluster_pref_normalized.index)
        n_rows = (n_clusters + 2) // 3  # 每行最多3个图
        for i, cluster in enumerate(cluster_pref_normalized.index):
            ax = plt.subplot(n_rows, 3, i+1, polar=True)
            
            # 准备雷达图数据
            values = cluster_pref_normalized.loc[cluster].tolist()
            values += values[:1]  # 闭合雷达图
            
            # 绘制雷达图
            ax.plot(angles, values, 'o-', linewidth=2, label=f'聚类 {cluster}')
            ax.fill(angles, values, alpha=0.25)
            
            # 设置雷达图属性
            ax.set_thetagrids(np.degrees(angles[:-1]), feature_factors)
            ax.set_title(f'聚类 {cluster} 用户偏好')
            ax.grid(True)
        
        plt.tight_layout()
        plt.savefig('results/bert_user_preference_radar.png', dpi=300)
        
        # 生成用户偏好分析报告
        with open('results/bert_preference_analysis.md', 'w', encoding='utf-8') as f:
            f.write("# BERT用户偏好分析报告\n\n")
            
            f.write("## 1. 用户评论聚类分析\n\n")
            f.write("基于BERT语义嵌入的评论聚类，将用户分为以下几个偏好群体：\n\n")
            
            # 计算每个聚类的大小和情感比例
            cluster_stats = df_with_embeddings.groupby('cluster').agg({
                'sentiment': [lambda x: len(x), lambda x: (x == 1).mean() * 100]
            })
            cluster_stats.columns = ['评论数量', '正面评论比例(%)']
            
            f.write(cluster_stats.to_markdown() + "\n\n")
            
            f.write("## 2. 用户群体偏好特征\n\n")
            f.write("不同用户群体对产品特征的偏好程度：\n\n")
            f.write(cluster_pref_df.to_markdown() + "\n\n")
            
            # 为每个产品分析主要用户群体
            product_cluster = pd.crosstab(
                df_with_embeddings['product'], 
                df_with_embeddings['cluster'], 
                normalize='index'
            ) * 100
            
            f.write("## 3. 产品与用户群体关系\n\n")
            f.write("各产品用户在不同偏好群体中的分布(%)：\n\n")
            f.write(product_cluster.to_markdown() + "\n\n")
            
            f.write("## 4. 产品优化建议\n\n")
            for product in df['product'].unique():
                f.write(f"### {product}\n\n")
                
                if product in product_cluster.index:
                    # 找出该产品最主要的用户群体
                    main_cluster = product_cluster.loc[product].idxmax()
                    main_pct = product_cluster.loc[product, main_cluster]
                    
                    # 找出该群体最在意的特征（如果没有特征得分，则使用所有产品评论计算）
                    if main_cluster in cluster_pref_df.index and cluster_pref_df.loc[main_cluster].max() > 0:
                        top_features = cluster_pref_df.loc[main_cluster].nlargest(2).index.tolist()
                        bottom_features = cluster_pref_df.loc[main_cluster].nsmallest(2).index.tolist()
                        
                        f.write(f"- **主要用户群体**: 聚类 {main_cluster} (占比: {main_pct:.1f}%)\n")
                        f.write(f"- **核心关注特征**: {', '.join(top_features)}\n")
                        f.write(f"- **建议**: 强化{top_features[0]}方面的体验，提升{bottom_features[0]}特性\n\n")
                        
                        # 根据不同特征提供具体建议
                        if top_features[0] == '性价比':
                            f.write("  - **价格策略**: 考虑优化定价或提供促销活动，突出产品性价比\n")
                        elif top_features[0] == '质量':
                            f.write("  - **品质保证**: 在产品宣传中强调高品质工艺和可靠性\n")
                        elif top_features[0] == '购物体验':
                            f.write("  - **服务优化**: 提升物流、包装和客服质量，改善整体购物体验\n")
                        elif top_features[0] == '实用性':
                            f.write("  - **功能宣传**: 重点展示产品的实用功能和便捷操作方式\n")
                    else:
                        # 如果没有有效特征得分，提供通用建议
                        f.write(f"- **主要用户群体**: 聚类 {main_cluster} (占比: {main_pct:.1f}%)\n")
                        f.write("- **建议**: 基于整体数据分析，建议全面提升产品质量与购物体验\n\n")
        
        print("BERT用户偏好分析完成，结果已保存")
        
        return {
            'embeddings': embeddings,
            'cluster_preferences': cluster_pref_df,
            'user_clusters': df_with_embeddings['cluster'],
            'visualization_path': 'results/bert_comment_embeddings.png',
            'report_path': 'results/bert_preference_analysis.md'
        }
        
    except Exception as e:
        import traceback
        print(f"BERT用户偏好分析失败: {str(e)}")
        traceback.print_exc()
        return None