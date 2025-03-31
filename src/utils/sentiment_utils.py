"""情感词典加载相关工具"""
import os
import traceback
from .file_utils import detect_file_encoding

def load_custom_sentiment_dicts(sentiment_dir):
    """增强的情感词典加载，添加格式验证"""
    positive_words = set()
    negative_words = set()
    
    if not os.path.exists(sentiment_dir):
        raise FileNotFoundError(f"情感词典目录不存在: {sentiment_dir}")

    for file_name in os.listdir(sentiment_dir):
        file_path = os.path.join(sentiment_dir, file_name)
        if file_name.endswith(".txt") and os.path.isfile(file_path):
            print(f"正在加载文件: {file_path}")
            try:
                encoding = detect_file_encoding(file_path)
                with open(file_path, "r", encoding=encoding, errors="ignore") as f:
                    lines = [line.strip() for line in f if line.strip()]
                    
                    # 验证词典内容
                    if not lines:
                        print(f"警告: 文件 {file_name} 为空")
                        continue
                        
                    if "正面" in file_name:
                        positive_words.update(lines)
                    elif "负面" in file_name:
                        negative_words.update(lines)
            except Exception as e:
                print(f"加载文件 {file_name} 时出错: {e}\n{traceback.format_exc()}")

    print(f"正面词语数量: {len(positive_words)}, 负面词语数量: {len(negative_words)}")
    return positive_words, negative_words
