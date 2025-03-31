import os
import pandas as pd
from tqdm import tqdm
import glob
import traceback
from openpyxl import load_workbook

# 从拆分的模块中导入函数
from utils.file_utils import detect_file_encoding
from utils.sentiment_utils import load_custom_sentiment_dicts
from utils.data_utils import safe_read_file
from analysis.sentiment_analysis import analyze_sentiment_with_custom_dict

def process_file(input_file, output_dir, text_column, positive_words, negative_words):
    """增强的文件处理函数，添加更多验证和错误处理"""
    # 检测文件编码
    encoding = detect_file_encoding(input_file)
    print(f"检测到文件 {input_file} 的编码为: {encoding}")

    # 读取数据
    df = safe_read_file(input_file)
    if df is None or df.empty:
        print(f"文件 {input_file} 读取失败或为空")
        return None

    # 检查文本列
    if text_column not in df.columns:
        print(f"文件 {os.path.basename(input_file)} 缺少列：'{text_column}'")
        return None
    
    # 数据清洗
    df[text_column] = df[text_column].fillna("").astype(str)
    
    # 过滤空文本
    df = df[df[text_column].str.strip() != ""]
    if df.empty:
        print(f"文件 {os.path.basename(input_file)} 无有效文本")
        return None

    # 情感分析
    try:
        tqdm.pandas(desc=f"分析 {os.path.basename(input_file)}")
        df['sentiment_score'] = df[text_column].progress_apply(
            lambda text: analyze_sentiment_with_custom_dict(text, positive_words, negative_words)
        )
    except Exception as e:
        print(f"情感分析过程中出错: {e}")
        return None
    
    # 保存结果
    output_file = os.path.join(output_dir, f"sentiment_{os.path.basename(input_file)}")
    try:
        if input_file.endswith('.xlsx'):
            df.to_excel(output_file, index=False, engine='openpyxl')
        else:
            df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"结果已保存：{output_file}")
        return output_file
    except Exception as e:
        print(f"保存结果文件失败: {e}")
        return None

def calculate_emotion_proportions(input_file):
    """增强的统计函数，添加除零保护"""
    try:
        if input_file.endswith('.xlsx'):
            df = pd.read_excel(input_file, engine='openpyxl')
        else:
            df = pd.read_csv(input_file, encoding='utf-8')
            
        if df.empty or 'sentiment_score' not in df.columns:
            raise ValueError("无效的数据文件")
            
        total = len(df)
        if total == 0:
            return {
                "file": os.path.basename(input_file),
                "positive_ratio": "0.00%",
                "neutral_ratio": "0.00%",
                "negative_ratio": "0.00%",
                "total_count": 0
            }
            
        positive = (df['sentiment_score'] > 0.1).sum()  # 使用阈值避免边缘值
        neutral = ((df['sentiment_score'] >= -0.1) & (df['sentiment_score'] <= 0.1)).sum()
        negative = (df['sentiment_score'] < -0.1).sum()
        
        return {
            "file": os.path.basename(input_file),
            "positive_ratio": f"{positive / total * 100:.2f}%",
            "neutral_ratio": f"{neutral / total * 100:.2f}%",
            "negative_ratio": f"{negative / total * 100:.2f}%",
            "total_count": total
        }
    except Exception as e:
        print(f"统计文件 {os.path.basename(input_file)} 时出错: {e}")
        return None

def batch_process(input_dir, output_dir, text_column, positive_words, negative_words):
    """增强的批量处理函数，添加更多日志和错误处理"""
    os.makedirs(output_dir, exist_ok=True)
    files = glob.glob(os.path.join(input_dir, '*.xlsx')) + glob.glob(os.path.join(input_dir, '*.csv'))
    
    if not files:
        raise ValueError(f"目录 {input_dir} 下未找到.xlsx或.csv文件")
    
    all_results = []
    for file in tqdm(files, desc="处理文件中"):
        try:
            print(f"\n正在处理文件: {os.path.basename(file)}")
            output_file = process_file(file, output_dir, text_column, positive_words, negative_words)
            if output_file:
                stats = calculate_emotion_proportions(output_file)
                if stats:
                    all_results.append(stats)
        except Exception as e:
            print(f"处理文件 {os.path.basename(file)} 时发生严重错误: {e}\n{traceback.format_exc()}")
    
    # 保存汇总结果
    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_file = os.path.join(output_dir, "summary_results.xlsx")
        try:
            summary_df.to_excel(summary_file, index=False, engine='openpyxl')
            print(f"\n汇总统计结果已保存：{summary_file}")
        except Exception as e:
            print(f"保存汇总结果失败: {e}")
    
    return all_results

if __name__ == "__main__":
    try:
        sentiment_dir = "sentiment"
        input_dir = "data/segment_results"
        output_dir = "data/statistic_results"
        text_column = "comment"

        print("正在加载情感词典...")
        positive_words, negative_words = load_custom_sentiment_dicts(sentiment_dir)
        
        print("\n开始批量处理文件...")
        results = batch_process(input_dir, output_dir, text_column, positive_words, negative_words)
        
        print("\n处理完成，统计结果:")
        for r in results:
            print(f"\n文件 {r['file']}:")
            print(f"正面: {r['positive_ratio']} 中性: {r['neutral_ratio']} 负面: {r['negative_ratio']}")
            print(f"总条数: {r['total_count']}")
    except Exception as e:
        print(f"程序运行出错: {e}\n{traceback.format_exc()}")
