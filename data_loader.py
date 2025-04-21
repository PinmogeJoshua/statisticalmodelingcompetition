import os
import pandas as pd
import re
import jieba
import numpy as np

def load_sentiment_data(data_dir="data/情感分类结果"):
    """加载情感分类结果数据"""
    # 存储所有产品数据
    all_data = []
    
    # 遍历情感分类结果文件夹
    for file in os.listdir(data_dir):
        if file.endswith("正面情感结果.txt"):
            product_name = file.replace("正面情感结果.txt", "")
            sentiment = 1  # 正面情感
        elif file.endswith("负面情感结果.txt"):
            product_name = file.replace("负面情感结果.txt", "")
            sentiment = -1  # 负面情感
        else:
            continue
        
        file_path = os.path.join(data_dir, file)
        
        # 读取文件内容，使用多种编码方式尝试
        encodings = ['utf-8', 'gbk', 'gb18030', 'latin1']
        content = None
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                    content = f.readlines()
                # 检查内容是否为中文
                if content and any('\u4e00' <= char <= '\u9fff' for char in ''.join(content[:5])):
                    break  # 找到了合适的编码
            except UnicodeDecodeError:
                continue
                
        if not content:
            print(f"警告: 无法正确读取文件 {file}")
            continue
        
        # 处理每行数据
        for line in content:
            line = line.strip()
            if not line:
                continue
                
            parts = line.split('\t', 1)  # 最多分割一次
            if len(parts) == 2:
                score = parts[0].strip()
                comment = parts[1].strip()
                
                if comment and len(comment) > 1:  # 确保评论不为空且有实际内容
                    all_data.append({
                        'product': product_name,
                        'sentiment': sentiment,
                        'score': score,
                        'comment': comment
                    })
    
    # 转换为DataFrame
    df = pd.DataFrame(all_data)
    print(f"成功加载了 {len(df)} 条评论数据")
    return df

def preprocess_text(df):
    """预处理文本数据，根据特征因素表提取特征"""
    print("开始文本预处理和特征提取...")
    
    # 特征因素表
    feature_keywords = {
        '性价比': ['值', '满意', '性价比', '价格', '便宜', '实惠', '划算', '贵', '优惠', '便宜', '值得', '值'],
        '质量': ['质量', '不错', '结实', '轻便', '好', '扎实', '稳', '轻', '牢固', '耐用', '品质', '做工'],
        '购物体验': ['物流', '发货', '客服', '服务', '快递', '包装', '态度', '周到', '收到', '配送', '送货', '到货'],
        '实用性': ['好用', '实用', '方便', '使用', '舒适', '轻便', '易用', '操作', '舒服', '方便', '灵活']
    }
    
    # 加载停用词
    stop_words = set()
    try:
        with open('cn_stopwords.txt', 'r', encoding='utf-8') as f:
            stop_words = set([line.strip() for line in f])
    except:
        print("无法加载停用词文件，将使用空停用词表")
    
    # 添加基本特征
    df['comment_length'] = df['comment'].apply(len)
    
    # 分词处理
    print("进行中文分词...")
    df['tokens'] = df['comment'].apply(lambda x: 
                                     [word for word in jieba.lcut(str(x)) 
                                      if word not in stop_words and len(word.strip()) > 0])
    
    # 提取各维度特征
    for factor, keywords in feature_keywords.items():
        print(f"提取 {factor} 特征...")
        
        # 计算每条评论中包含该特征维度关键词的数量
        df[f'{factor}_count'] = df['tokens'].apply(
            lambda tokens: sum(1 for token in tokens if token in keywords)
        )
        
        # 根据情感标签调整特征得分
        df[f'{factor}_score'] = df[f'{factor}_count'] * df['sentiment']
        
        # 判断评论是否提及该特征
        df[f'has_{factor}'] = df[f'{factor}_count'].apply(lambda x: 1 if x > 0 else 0)
        
        # 打印统计信息
        mention_count = df[f'has_{factor}'].sum()
        mention_rate = mention_count / len(df) * 100
        print(f"- {factor} 被提及 {mention_count} 次，提及率 {mention_rate:.2f}%")
    
    print("文本预处理完成")
    return df

# 在 preprocess_text 函数后添加：

def enhanced_feature_engineering(df):
    """增强特征工程，添加交互特征和多项式特征"""
    print("执行增强特征工程...")
    
    df_engineered = df.copy()
    
    # 1. 添加交互特征
    for i, feat1 in enumerate(['性价比', '质量', '购物体验', '实用性']):
        for j, feat2 in enumerate(['性价比', '质量', '购物体验', '实用性']):
            if i < j:  # 避免重复
                # 添加计数和得分的交互特征
                df_engineered[f'{feat1}_{feat2}_count_interact'] = df_engineered[f'{feat1}_count'] * df_engineered[f'{feat2}_count']
                df_engineered[f'{feat1}_{feat2}_score_interact'] = df_engineered[f'{feat1}_score'] * df_engineered[f'{feat2}_score']
                print(f"- 创建交互特征: {feat1}_{feat2}")
    
    # 2. 添加多项式特征
    for feat in ['性价比', '质量', '购物体验', '实用性']:
        df_engineered[f'{feat}_count_squared'] = df_engineered[f'{feat}_count'] ** 2
        df_engineered[f'{feat}_score_squared'] = df_engineered[f'{feat}_score'] ** 2
        print(f"- 创建多项式特征: {feat} 平方项")
    
    # 3. 添加评论长度的归一化和划分特征
    # 评论长度分段
    df_engineered['comment_length_norm'] = (df_engineered['comment_length'] - df_engineered['comment_length'].mean()) / df_engineered['comment_length'].std()
    
    # 将评论长度分为短、中、长三类
    length_bins = [0, 50, 200, float('inf')]
    length_labels = ['short', 'medium', 'long']
    df_engineered['comment_length_category'] = pd.cut(df_engineered['comment_length'], bins=length_bins, labels=length_labels)
    
    # 将分类转换为独热编码
    comment_length_dummies = pd.get_dummies(df_engineered['comment_length_category'], prefix='comment_length')
    df_engineered = pd.concat([df_engineered, comment_length_dummies], axis=1)
    print("- 创建评论长度分类特征")
    
    # 4. 添加特征组合比率
    for feat in ['性价比', '质量', '购物体验', '实用性']:
        # 计算特征得分与计数的比率
        df_engineered[f'{feat}_score_count_ratio'] = df_engineered[f'{feat}_score'] / (df_engineered[f'{feat}_count'] + 1)  # 加1避免除零
        print(f"- 创建比率特征: {feat}_score_count_ratio")
    
    print(f"增强特征工程完成，从{len(df.columns)}个特征扩展到{len(df_engineered.columns)}个特征")
    
    # 5. 处理极端值和异常值
    numeric_cols = df_engineered.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        # 计算第1和第99百分位数
        q1 = df_engineered[col].quantile(0.01)
        q99 = df_engineered[col].quantile(0.99)
        
        # 限制极端值
        if col != 'sentiment':  # 不处理目标变量
            df_engineered[col] = df_engineered[col].clip(q1, q99)
            print(f"- 处理特征 {col} 的极端值")
    
    # 6. 添加聚类特征
    from sklearn.cluster import KMeans
    
    # 选择用于聚类的数值特征
    cluster_features = ['性价比_score', '质量_score', '购物体验_score', '实用性_score']
    
    if all(feat in df_engineered.columns for feat in cluster_features):
        # 进行K-means聚类
        print("- 添加聚类特征")
        kmeans = KMeans(n_clusters=3, random_state=42)
        df_engineered['cluster'] = kmeans.fit_predict(df_engineered[cluster_features])
        
        # 转换为独热编码
        cluster_dummies = pd.get_dummies(df_engineered['cluster'], prefix='cluster')
        df_engineered = pd.concat([df_engineered, cluster_dummies], axis=1)
    
    return df_engineered

def get_feature_importance_data():
    """获取特征因素数据"""
    # 表3: 新特征因素表
    feature_factors = {
        '性价比': ['值', '满意', '性价比', '价格'],
        '质量': ['质量', '不错', '结实', '轻便'],
        '购物体验': ['物流', '发货', '客服', '服务'],
        '实用性': ['好用', '实用', '方便']
    }
    
    # 表2: 八款产品高频词汇统计表（部分数据）
    high_freq_words = pd.DataFrame({
        '关键词': ['值', '满意', '好用', '实用', '方便', '质量', '不错', '结实', '轻便', '物流', '发货', '客服', '服务', '性价比', '价格'],
        '中星众科': [84, 65, 93, 50, 196, 261, 174, 82, 19, 80, 56, 62, 71, 12, 24],
        '可孚': [38, 25, 56, 22, 62, 98, 82, 20, 34, 33, 13, 13, 15, 4, 7],
        '耀典': [21, 22, 27, 8, 28, 48, 48, 10, 5, 12, 11, 4, 11, 6, 4],
        '雅德': [5, 5, 5, 0, 7, 11, 5, 4, 3, 5, 4, 8, 6, 1, 0],
        '鱼跃': [38, 25, 56, 22, 62, 98, 82, 20, 34, 33, 13, 13, 15, 4, 7],
        '阿里健康大药房': [53, 51, 96, 46, 116, 175, 157, 55, 69, 45, 45, 29, 21, 12, 20],
        '福仕德': [22, 25, 43, 14, 56, 87, 64, 24, 16, 23, 20, 30, 24, 4, 10],
        'Hfine': [25, 27, 36, 19, 58, 96, 75, 32, 52, 20, 16, 15, 17, 8, 9]
    })
    
    return {
        'feature_factors': feature_factors,
        'high_freq_words': high_freq_words
    }
    