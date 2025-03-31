import os
import sys
import pandas as pd
from .construct_mood_dict import analyze_sentiment
from .statistics import calculate_emotion_proportions

# 添加项目根目录到 sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)
print("项目根目录已添加到 sys.path:", project_root)

# 定义输入和输出文件夹
input_dir = os.path.abspath(os.path.join(project_root, "data/segment_results"))
output_dir = os.path.abspath(os.path.join(project_root, "data/results"))

# 检查输入文件夹是否存在
if not os.path.exists(input_dir):
    raise FileNotFoundError(f"输入文件夹不存在: {input_dir}")

# 如果输出文件夹不存在，则创建
os.makedirs(output_dir, exist_ok=True)

# 获取所有 .xlsx 文件
input_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".xlsx")]

# 遍历每个文件并进行情感分析
for file in input_files:
    input_path = os.path.join(input_dir, file)
    output_path = os.path.join(output_dir, f"{os.path.splitext(file)[0]}_sentiment.xlsx")
    
    print(f"正在处理文件: {input_path}")
    analyze_sentiment(input_path, output_path)

print("所有文件的情感分析已完成！结果保存在:", output_dir)

# 定义一个列表用于存储所有文件的统计结果
all_stats = []

# 统计情绪比例
for file in input_files:
    input_path = os.path.join(input_dir, file)
    output_path = os.path.join(output_dir, f"{os.path.splitext(file)[0]}_sentiment.xlsx")
    
    print(f"正在处理文件: {input_path}")
    analyze_sentiment(input_path, output_path)

    # 调用统计功能
    stats = calculate_emotion_proportions(output_path)
    stats['file_name'] = file  # 添加文件名到统计结果
    all_stats.append(stats)  # 将结果添加到列表中
    print(f"文件 {file} 的情绪比例统计结果：", stats)

# 将所有统计结果保存到一个 Excel 文件
stats_output_path = os.path.join(output_dir, "emotion_statistics_summary.xlsx")
df_stats = pd.DataFrame(all_stats)
df_stats.to_excel(stats_output_path, index=False)

print(f"所有情绪比例统计结果已保存到：{stats_output_path}")