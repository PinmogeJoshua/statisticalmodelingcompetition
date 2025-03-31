import os
import pandas as pd
from snownlp import SnowNLP
from tqdm import tqdm
import glob  # 确保正确导入glob模块

def analyze_sentiment(text):
    """分析单条文本情感得分（0-1 → 映射为-1到1）"""
    try:
        return SnowNLP(text).sentiments * 2 - 1  # 转换为[-1,1]区间
    except:
        return 0.0  # 解析失败时返回中性值0

def process_file(input_file, output_dir, text_column='content'):
    """
    处理单个文件：读取 → 情感分析 → 保存结果
    :param input_file:   输入文件路径（.xlsx/.csv）
    :param output_dir:   输出目录路径
    :param text_column:  文本列名
    """
    # 读取数据
    df = pd.read_excel(input_file) if input_file.endswith('.xlsx') else pd.read_csv(input_file)
    
    # 检查文本列
    if text_column not in df.columns:
        raise ValueError(f"文件 {os.path.basename(input_file)} 缺少列：'{text_column}'")
    
    # 情感分析（显示进度条）
    tqdm.pandas(desc=f"分析 {os.path.basename(input_file)}")
    df['sentiment_score'] = df[text_column].progress_apply(analyze_sentiment)
    
    # 保存结果
    output_file = os.path.join(output_dir, f"sentiment_{os.path.basename(input_file)}")
    df.to_excel(output_file, index=False)
    print(f"结果已保存：{output_file}")
    return output_file  # 返回结果文件路径

def calculate_emotion_proportions(input_file):
    """统计单个文件的情感比例"""
    df = pd.read_excel(input_file) if input_file.endswith('.xlsx') else pd.read_csv(input_file)
    
    if 'sentiment_score' not in df.columns:
        raise ValueError(f"文件 {os.path.basename(input_file)} 缺少情感得分列")
    
    total = len(df)
    positive = (df['sentiment_score'] > 0).sum()
    neutral = (df['sentiment_score'] == 0).sum()
    negative = (df['sentiment_score'] < 0).sum()
    
    return {
        "file": os.path.basename(input_file),
        "positive_ratio": f"{positive / total * 100:.2f}%",
        "neutral_ratio": f"{neutral / total * 100:.2f}%",
        "negative_ratio": f"{negative / total * 100:.2f}%",
        "total_count": total
    }

def batch_process(input_dir, output_dir, text_column='content'):
    """
    批量处理目录下的所有文件
    :param input_dir:   输入目录路径
    :param output_dir:  输出目录路径
    :param text_column: 文本列名
    :return: 所有文件的统计结果列表
    """
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"输入目录不存在：{input_dir}")
    os.makedirs(output_dir, exist_ok=True)  # 创建输出目录
    
    # 获取所有.xlsx和.csv文件（使用glob.glob）
    files = []
    for ext in ['*.xlsx', '*.csv']:
        files.extend(glob.glob(os.path.join(input_dir, ext)))  # 修正为glob.glob
    
    if not files:
        raise ValueError(f"目录 {input_dir} 下未找到.xlsx或.csv文件")
    
    # 逐个处理文件
    all_results = []
    for file in tqdm(files, desc="处理文件中"):
        try:
            output_file = process_file(file, output_dir, text_column)
            stats = calculate_emotion_proportions(output_file)
            all_results.append(stats)
        except Exception as e:
            print(f"处理文件 {os.path.basename(file)} 失败：{str(e)}")
    
    # 保存汇总统计结果
    summary_df = pd.DataFrame(all_results)
    summary_file = os.path.join(output_dir, "summary_results.xlsx")
    summary_df.to_excel(summary_file, index=False)
    print(f"\n汇总统计结果已保存：{summary_file}")
    return all_results

# 示例调用
if __name__ == "__main__":
    input_dir = "data/segment_results"   # 输入目录（包含多个.xlsx/.csv文件）
    output_dir = "data/statistic_results"         # 输出目录
    text_column = "comment"            # 你的文本列名
    
    # 批量处理并打印结果
    results = batch_process(input_dir, output_dir, text_column)
    for r in results:
        print(f"\n文件 {r['file']} 统计结果：")
        print(f"正面: {r['positive_ratio']}, 中性: {r['neutral_ratio']}, 负面: {r['negative_ratio']}")
        print(f"总条数: {r['total_count']}")
    