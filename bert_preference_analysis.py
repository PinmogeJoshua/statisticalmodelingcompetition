import torch
from transformers import BertTokenizer, BertModel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import os

def setup_bert_model():
    """加载BERT模型和分词器"""
    print("加载BERT模型...")
    # 使用中文BERT预训练模型
    model_name = "bert-base-chinese"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    return tokenizer, model

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
        # 加载BERT模型
        tokenizer, model = setup_bert_model()
        
        # 提取评论嵌入
        embeddings = extract_bert_embeddings(df, tokenizer, model)
        
        # 将评论嵌入与产品信息关联
        df_with_embeddings = df.copy()
        
        # 使用降维技术将高维BERT嵌入降到2维进行可视化
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
        
        # 使用PCA初步降维以加速t-SNE
        pca = PCA(n_components=50)
        reduced_embeddings = pca.fit_transform(embeddings)
        
        # 使用t-SNE进一步降维到2维
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embeddings_2d = tsne.fit_transform(reduced_embeddings)
        
        # 添加2D嵌入到DataFrame
        df_with_embeddings['embedding_x'] = embeddings_2d[:, 0]
        df_with_embeddings['embedding_y'] = embeddings_2d[:, 1]
        
        # 可视化评论嵌入空间，按产品和情感分类
        plt.figure(figsize=(12, 10))
        
        # 为不同产品分配不同颜色
        products = df['product'].unique()
        product_colors = dict(zip(products, plt.get_cmap('tab10')(np.linspace(0, 1, len(products)))))
        
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
        kmeans = KMeans(n_clusters=5, random_state=42)
        df_with_embeddings['cluster'] = kmeans.fit_predict(embeddings)
        
        # 计算每个聚类的特征偏好得分
        feature_factors = ['性价比', '质量', '购物体验', '实用性']
        cluster_preferences = {}
        
        for cluster in df_with_embeddings['cluster'].unique():
            cluster_df = df_with_embeddings[df_with_embeddings['cluster'] == cluster]
            cluster_preferences[cluster] = {}
            
            for factor in feature_factors:
                factor_score_col = f'{factor}_score'
                if factor_score_col in cluster_df.columns:
                    # 计算该聚类中此特征的平均得分
                    avg_score = cluster_df[factor_score_col].mean()
                else:
                    # 如果没有特征得分列，可能是因为没有通过has_*列提取特征
                    # 使用提及次数作为替代
                    has_factor_col = f'has_{factor}'
                    factor_count_col = f'{factor}_count'
                    
                    if has_factor_col in cluster_df.columns:
                        mention_rate = cluster_df[has_factor_col].mean()
                        avg_score = mention_rate * 2  # 简单放大，使得范围在0-2之间
                    elif factor_count_col in cluster_df.columns:
                        avg_count = cluster_df[factor_count_col].mean()
                        avg_score = avg_count  
                    else:
                        avg_score = 0.0
                
                cluster_preferences[cluster][factor] = avg_score
        
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
        for i, cluster in enumerate(cluster_pref_normalized.index):
            ax = plt.subplot(2, 3, i+1, polar=True)
            
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
            
            f.write("## 3. 用户偏好洞察\n\n")
            
            # 为每个聚类提供一些洞察
            for cluster in cluster_pref_df.index:
                f.write(f"### 聚类 {cluster} 用户群体\n\n")
                
                # 找出该聚类最偏好和最不偏好的特征
                pref_series = cluster_pref_df.loc[cluster]
                most_pref = pref_series.nlargest(2)
                least_pref = pref_series.nsmallest(2)
                
                f.write(f"- **群体规模**: {cluster_stats.loc[cluster, '评论数量']} 条评论\n")
                f.write(f"- **情感倾向**: {cluster_stats.loc[cluster, '正面评论比例(%)']:.2f}% 正面评论\n")
                f.write(f"- **最关注特征**: {most_pref.index.tolist()[0]} (得分: {most_pref.values[0]:.2f}), {most_pref.index.tolist()[1]} (得分: {most_pref.values[1]:.2f})\n")
                f.write(f"- **最不关注特征**: {least_pref.index.tolist()[0]} (得分: {least_pref.values[0]:.2f}), {least_pref.index.tolist()[1]} (得分: {least_pref.values[1]:.2f})\n\n")
                
                # 分析该聚类的产品偏好
                cluster_product_dist = df_with_embeddings[df_with_embeddings['cluster'] == cluster]['product'].value_counts(normalize=True) * 100
                f.write("**产品偏好分布**:\n\n")
                for product, pct in cluster_product_dist.items():
                    f.write(f"- {product}: {pct:.2f}%\n")
                f.write("\n")
            
            f.write("## 4. 产品推荐策略\n\n")
            f.write("基于用户偏好分析，可以制定以下产品推荐策略：\n\n")
            
            # 为每个产品提供一些基于用户偏好的推荐策略
            for product in df['product'].unique():
                f.write(f"### {product}\n\n")
                
                # 找出对该产品评论最多的聚类
                product_clusters = df_with_embeddings[df_with_embeddings['product'] == product]['cluster'].value_counts()
                main_cluster = product_clusters.index[0]
                
                f.write(f"- **主要用户群体**: 聚类 {main_cluster} (占比: {product_clusters[main_cluster]/product_clusters.sum()*100:.2f}%)\n")
                f.write(f"- **用户偏好特征**: {cluster_pref_df.loc[main_cluster].nlargest(2).index.tolist()}\n")
                f.write(f"- **推荐策略**: 根据该用户群体的特征偏好，在产品营销中强调其{cluster_pref_df.loc[main_cluster].nlargest(1).index.tolist()[0]}和{cluster_pref_df.loc[main_cluster].nlargest(2).index.tolist()[1]}特性\n\n")
        
        print("BERT用户偏好分析完成，结果保存至'results/bert_preference_analysis.md'")
        
        return {
            'embeddings': embeddings,
            'cluster_preferences': cluster_pref_df,
            'user_clusters': df_with_embeddings['cluster']
        }
        
    except Exception as e:
        print(f"BERT用户偏好分析失败: {str(e)}")
        return None