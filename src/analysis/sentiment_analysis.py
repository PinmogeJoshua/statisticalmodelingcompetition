from snownlp import SnowNLP

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