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
    