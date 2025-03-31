import pandas as pd

def calculate_emotion_proportions(input_path):
    """
    基于情感得分统计情绪所占比例
    :param input_path: 情感分析结果的 Excel 文件路径
    :return: 一个字典，包含各情绪的比例和总发言数
    """
    # 读取情感分析结果
    df = pd.read_excel(input_path)

    # 确保包含得分列
    if 'sentiment_score' not in df.columns:
        raise ValueError("输入文件缺少必要的列：'sentiment_score'")

    # 根据得分分类情绪
    df['positive'] = df['sentiment_score'] > 0
    df['neutral'] = df['sentiment_score'] == 0
    df['negative'] = df['sentiment_score'] < 0

    # 统计各情绪的数量
    total_count = len(df)
    positive_count = df['positive'].sum()
    neutral_count = df['neutral'].sum()
    negative_count = df['negative'].sum()

    # 计算比例
    positive_ratio = positive_count / total_count * 100
    neutral_ratio = neutral_count / total_count * 100
    negative_ratio = negative_count / total_count * 100

    # 返回结果
    return {
        "positive_ratio": f"{positive_ratio:.2f}%",
        "neutral_ratio": f"{neutral_ratio:.2f}%",
        "negative_ratio": f"{negative_ratio:.2f}%",
        "total_count": total_count
    }

# 示例调用
if __name__ == "__main__":
    input_file = "data/results/sentiment_analysis.xlsx"  # 替换为实际路径
    result = calculate_emotion_proportions(input_file)
    print("情绪比例统计结果：", result)
    