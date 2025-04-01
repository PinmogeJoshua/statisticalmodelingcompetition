import re
import pandas as pd
import numpy as np
from snownlp import SnowNLP
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# 定义评价维度和相关关键词
DIMENSIONS = {
    "安全性": ["安全", "稳定", "防滑", "牢固", "支撑", "可靠", "防摔", "耐用"],
    "舒适性": ["舒适", "柔软", "贴合", "轻便", "顺滑", "无噪音", "缓冲"],
    "操作便捷性": ["操作", "简单", "方便", "易用", "灵活", "调节", "折叠", "安装"],
    "外观设计": ["外观", "设计", "颜色", "造型", "美观", "时尚", "轻巧"],
    "性价比": ["价格", "实惠", "划算", "性价比", "贵", "便宜", "值得"],
    "售后服务": ["售后", "客服", "服务", "维修", "响应", "退换", "态度"]
}

def analyze_sentiment_with_custom_dict(text, positive_words, negative_words):
    """增强的情感分析函数，添加更多边界条件处理"""
    try:
        if not isinstance(text, str) or not text.strip():
            return 0.0
            
        # 添加长度检查
        if len(text) > 10000:  # 处理超长文本
            text = text[:10000]
            
        try:
            s = SnowNLP(text)
            words = s.words
        except:
            # 如果分词失败，使用简单分词
            words = text.split()
            
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        total_count = positive_count + negative_count

        # 更安全的比例计算
        if total_count == 0:
            # 无情感词时，使用SnowNLP的默认情感分析
            try:
                return SnowNLP(text).sentiments - 0.5  # 转换为-0.5到0.5范围
            except:
                return 0.0
                
        # 标准化到-1到1范围
        score = (positive_count - negative_count) / total_count
        return max(-1.0, min(1.0, score))  # 确保在范围内
    except Exception as e:
        print(f"情感分析失败: {e}\n文本: {text[:100]}...")
        return 0.0
    
def dimension_sentiment_analysis(text, dimension_keywords, expanded_positive_words, expanded_negative_words):
    """对特定维度进行情感分析
    
    Args:
        text: 输入文本
        dimension_keywords: 维度关键词字典
        expanded_positive_words: 扩展后的正面情感词典
        expanded_negative_words: 扩展后的负面情感词典
        
    Returns:
        各维度的情感得分字典
    """
    from src.analysis.construct_mood_dict import calculate_sentiment_score
    
    dimension_scores = {}
    
    # 在文本中查找与各维度相关的部分
    for dimension, keywords in dimension_keywords.items():
        dimension_text = ""
        
        # 提取包含维度关键词的句子
        sentences = re.split(r'[。！？.!?]', text)
        for sentence in sentences:
            if any(keyword in sentence for keyword in keywords):
                dimension_text += sentence + "。"
        
        # 对提取的文本进行情感分析
        if dimension_text:
            dimension_scores[dimension] = calculate_sentiment_score(
                dimension_text, expanded_positive_words, expanded_negative_words
            )
        else:
            dimension_scores[dimension] = None  # 无法评估该维度
    
    return dimension_scores

def analyze_review_dimensions(df, dimension_keywords, expanded_positive_words, expanded_negative_words):
    """对评论进行多维度情感分析
    
    Args:
        df: 包含评论的DataFrame
        dimension_keywords: 维度关键词字典
        expanded_positive_words: 扩展后的正面情感词典 
        expanded_negative_words: 扩展后的负面情感词典
        
    Returns:
        添加了维度情感得分的DataFrame
    """
    # 初始化结果列
    for dimension in dimension_keywords.keys():
        df[f'sentiment_{dimension}'] = None
    
    # 确认评论列存在
    comment_col = None
    for col in ['comment', '评论', '内容', 'Comment']:
        if col in df.columns:
            comment_col = col
            break
            
    if not comment_col:
        print(f"警告: 未找到评论列，无法进行多维度分析")
        return df
    
    # 对每条评论进行多维度分析
    for i, row in df.iterrows():
        if comment_col in row and isinstance(row[comment_col], str):
            dimension_scores = dimension_sentiment_analysis(
                row[comment_col], dimension_keywords, 
                expanded_positive_words, expanded_negative_words
            )
            
            # 更新数据框
            for dimension, score in dimension_scores.items():
                df.at[i, f'sentiment_{dimension}'] = score
    
    return df

def extract_topics_from_reviews(df, n_topics=5, n_top_words=10):
    """使用LDA从评论中提取主题
    
    Args:
        df: 包含评论的DataFrame
        n_topics: 主题数量
        n_top_words: 每个主题的关键词数量
        
    Returns:
        topics: 主题列表
        topic_sentiments: 主题情感得分字典
        df: 添加了主题标签的DataFrame
    """
    # 确认评论列存在
    comment_col = None
    for col in ['comment', '评论', '内容', 'Comment']:
        if col in df.columns:
            comment_col = col
            break
            
    if not comment_col:
        print(f"警告: 未找到评论列，无法进行主题提取")
        return [], {}, df
    
    # 准备文本数据
    documents = df[comment_col].fillna('').astype(str).tolist()
    
    # 特征提取
    tfidf_vectorizer = TfidfVectorizer(
        max_df=0.95, min_df=2, 
        stop_words=['的', '了', '是', '我', '有', '和', '就', '在', '不', '这']
    )
    tfidf = tfidf_vectorizer.fit_transform(documents)
    
    # LDA主题模型
    lda = LatentDirichletAllocation(
        n_components=n_topics, 
        max_iter=50,
        learning_method='online',
        random_state=42
    )
    lda.fit(tfidf)
    
    # 获取特征名（词语）
    feature_names = tfidf_vectorizer.get_feature_names_out()
    
    # 提取主题词
    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        top_words_idx = topic.argsort()[:-n_top_words - 1:-1]
        top_words = [feature_names[i] for i in top_words_idx]
        topics.append({
            'topic_id': topic_idx,
            'words': top_words
        })
    
    # 计算每篇评论的主题分布
    topic_distribution = lda.transform(tfidf)
    df['dominant_topic'] = topic_distribution.argmax(axis=1)
    
    # 计算每个主题的情感得分
    topic_sentiments = {}
    for topic_id in range(n_topics):
        topic_reviews = df[df['dominant_topic'] == topic_id]
        if not topic_reviews.empty and 'sentiment_score' in topic_reviews.columns:
            topic_sentiments[topic_id] = topic_reviews['sentiment_score'].mean()
    
    return topics, topic_sentiments, df
