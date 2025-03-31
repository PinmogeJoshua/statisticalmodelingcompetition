import os
import sys
from .construct_mood_dict import analyze_sentiment

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
