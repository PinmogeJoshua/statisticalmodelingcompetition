import os
from transformers import BertTokenizer, BertModel

def load_local_bert_model():
    """从本地加载BERT模型"""
    try:
        # 指定本地BERT模型路径
        model_path = "c:\\VSCode\\StatisticalModeling\\models\\bert-base-chinese"
        
        print(f"正在从本地路径加载BERT模型: {model_path}")
        
        # 检查路径是否存在
        if not os.path.exists(model_path):
            print(f"警告: 模型路径不存在: {model_path}")
            return None, None
            
        # 加载tokenizer和model
        tokenizer = BertTokenizer.from_pretrained(model_path)
        model = BertModel.from_pretrained(model_path)
        
        print("BERT模型加载成功!")
        return tokenizer, model
    
    except Exception as e:
        print(f"加载BERT模型失败: {str(e)}")
        return None, None

# 测试代码 - 直接运行此文件时执行
if __name__ == "__main__":
    tokenizer, model = load_local_bert_model()
    if tokenizer and model:
        print("成功加载模型和tokenizer")
        # 简单测试
        text = "这是一个测试句子"
        encoded = tokenizer(text, return_tensors="pt")
        print(f"Tokenized: {encoded}")
        print("模型加载测试成功!")
    else:
        print("模型加载失败")