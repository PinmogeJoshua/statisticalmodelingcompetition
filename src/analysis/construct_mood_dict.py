"""
模块名称：construct_mood_dict
功能：构建情感词典并进行情感分析
"""

import re
import pandas as pd
from gensim.models import Word2Vec

# 1. 手动构建初级情感词典
positive_words = {"好": 10, "棒": 8, "满意": 7}  # 正面情感词汇及权重
negative_words = {"差": -10, "糟糕": -8, "失望": -7}  # 负面情感词汇及权重
neutral_words = {"一般": 0, "还行": 0}  # 中性情感词汇

# 程度副词及权重
degree_words = {"非常": 2, "特别": 1.5, "稍微": 0.5}

# 否定词
negation_words = {"不", "没", "无"}

# 感叹词
exclamation_words = {"！", "?"}

# 2. 通过语义相似度扩展情感词典
def expand_emotion_dict(corpus, base_words, threshold=0.6):
    """
    使用语义相似度扩展情感词典
    :param corpus: 语料库（列表形式，每个元素为一个句子）
    :param base_words: 基准情感词典
    :param threshold: 相似度阈值
    :return: 扩展后的情感词典
    """
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

# 3. 计算评论的情感得分
def calculate_sentiment_score(text, expanded_positive_words, expanded_negative_words):
    """
    计算单条评论的情感得分
    :param text: 输入文本
    :param expanded_positive_words: 扩展后的正面情感词典
    :param expanded_negative_words: 扩展后的负面情感词典
    :return: 情感得分
    """
    words = re.findall(r'[\u4e00-\u9fa5]+', text)  # 提取中文词汇
    score = 0
    negation_count = 0  # 否定词计数

    for word in words:
        if word in expanded_positive_words:
            weight = expanded_positive_words[word]
            if negation_count % 2 == 1:  # 否定词数量为单数时反转情感
                weight = -weight
            score += weight
        elif word in expanded_negative_words:
            weight = expanded_negative_words[word]
            if negation_count % 2 == 1:  # 否定词数量为单数时反转情感
                weight = -weight
            score += weight
        elif word in degree_words:
            score *= degree_words[word]  # 调整情感强度
        elif word in negation_words:
            negation_count += 1

    return score

# 4. 对评论数据集进行情感分析
def analyze_sentiment(dataset_path, output_path):
    """
    对评论数据集进行情感分析
    :param dataset_path: 输入数据集路径
    :param output_path: 输出结果路径
    """
    df = pd.read_excel(dataset_path)
    if 'comment' not in df.columns:
        print("数据集中未找到 'comment' 列，请检查数据格式。")
        return

    # 示例语料库（需要根据实际情况替换）
    corpus = [["这个", "产品", "非常", "好"], ["服务", "很", "差"], ["质量", "一般"]]

    # 扩展情感词典
    expanded_positive_words = expand_emotion_dict(corpus, positive_words)
    expanded_negative_words = expand_emotion_dict(corpus, negative_words)

    # 计算每条评论的情感得分
    df['sentiment_score'] = df['comment'].apply(
        lambda x: calculate_sentiment_score(
            str(x), expanded_positive_words, expanded_negative_words
        )
    )

    # 保存结果
    df.to_excel(output_path, index=False)
    print(f"情感分析完成，结果已保存到：{output_path}")
    