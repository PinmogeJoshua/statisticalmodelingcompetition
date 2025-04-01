"""高级情感分析工具"""
import jieba
import pandas as pd
import traceback
from gensim.models import Word2Vec, KeyedVectors

def enhance_with_word_vectors(text, word_vectors, positive_seeds, negative_seeds, threshold=0.6):
    """使用词向量增强情感分析
    
    Args:
        text: 输入文本
        word_vectors: 预训练词向量模型
        positive_seeds: 正面种子词列表
        negative_seeds: 负面种子词列表
        threshold: 相似度阈值
        
    Returns:
        情感得分 (-1 到 1 范围)
    """
    words = list(jieba.cut(text))
    positive_score = 0
    negative_score = 0
    word_count = 0
    
    for word in words:
        if word in word_vectors:
            # 计算与正面种子词的相似度
            positive_similarities = [
                word_vectors.similarity(word, seed) 
                for seed in positive_seeds if seed in word_vectors
            ]
            # 计算与负面种子词的相似度
            negative_similarities = [
                word_vectors.similarity(word, seed) 
                for seed in negative_seeds if seed in word_vectors
            ]
            
            if positive_similarities and max(positive_similarities) > threshold:
                positive_score += max(positive_similarities)
                word_count += 1
            if negative_similarities and max(negative_similarities) > threshold:
                negative_score += max(negative_similarities)
                word_count += 1
    
    if word_count > 0:
        return (positive_score - negative_score) / word_count
    return 0.0

def expand_emotion_dict(corpus, base_words, threshold=0.6):
    """使用语义相似度扩展情感词典
    
    Args:
        corpus: 词语列表的列表
        base_words: 基准情感词典
        threshold: 相似度阈值
        
    Returns:
        扩展后的情感词典
    """
    try:
        # 训练 Word2Vec 模型
        model = Word2Vec(sentences=corpus, vector_size=100, window=5, min_count=1, workers=4)
        
        expanded_dict = base_words.copy()
        for word, weight in base_words.items():
            if word in model.wv:
                similar_words = model.wv.most_similar(word, topn=20)
                for similar_word, similarity in similar_words:
                    if similarity > threshold:
                        expanded_dict[similar_word] = weight
        return expanded_dict
    except Exception as e:
        print(f"扩展情感词典时出错: {str(e)}")
        traceback.print_exc()
        return base_words

def categorize_sentiment(score):
    """对情感得分进行细致分类
    
    Args:
        score: 情感得分 (-1 到 1 范围)
        
    Returns:
        情感分类标签
    """
    if score is None or pd.isna(score):
        return None
        
    if score >= 0.5:
        return '非常正面'
    elif 0.1 <= score < 0.5:
        return '正面'
    elif -0.1 <= score < 0.1:
        return '中性'
    elif -0.5 <= score < -0.1:
        return '负面'
    else:
        return '非常负面'

def load_pretrained_word_vectors(vector_path):
    """加载预训练的词向量模型
    
    Args:
        vector_path: 词向量文件路径
        
    Returns:
        词向量模型
    """
    try:
        return KeyedVectors.load_word2vec_format(vector_path, binary=False)
    except Exception as e:
        print(f"加载词向量模型时出错: {str(e)}")
        traceback.print_exc()
        return None