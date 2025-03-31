import os
import jieba
import pandas as pd
from tqdm import tqdm

# 文件夹配置
input_dir = "./compressed_results"    # 输入文件夹
output_dir = "./segment_results"      # 输出文件夹

# 列名配置
text_column = "字段1"     # 待分词的列名
seg_column = "分词结果"    # 分词结果列名

# 停用词过滤（可选）
stopwords = set()
# with open("stopwords.txt", "r", encoding="utf-8") as f:
#     stopwords = set(line.strip() for line in f)

def segment_text(text):
    """处理单个文本的分词和停用词过滤"""
    if pd.isna(text):
        return ""
    text = str(text).strip()
    if not text:
        return ""
    words = jieba.cut(text, cut_all=False)
    return " ".join(word for word in words if not stopwords or word not in stopwords)

def process_excel(input_path, output_path):
    """处理单个Excel文件"""
    try:
        df = pd.read_excel(input_path, engine='openpyxl')
    except Exception as e:
        print(f"错误：读取文件 {input_path} 失败 - {str(e)}")
        return

    if seg_column in df.columns:
        print(f"跳过已分词文件: {input_path}")
        return

    # 使用 tqdm 进度条
    tqdm.pandas(desc=f"处理 {os.path.basename(input_path)}")
    df[seg_column] = df[text_column].progress_apply(segment_text)

    try:
        os.makedirs(output_dir, exist_ok=True)
        df.to_excel(output_path, index=False, engine='openpyxl')
        print(f"保存成功: {output_path}")
    except Exception as e:
        print(f"错误：保存文件 {output_path} 失败 - {str(e)}")

if __name__ == '__main__':
    # 获取输入文件（严格匹配 .xlsx 后缀）
    input_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".xlsx")]
    if not input_files:
        print(f"错误：输入目录 {input_dir} 中未找到 .xlsx 文件！")
        exit(1)

    # 处理所有文件
    for file in input_files:
        input_path = os.path.join(input_dir, file)
        output_path = os.path.join(output_dir, f"{os.path.splitext(file)[0]}_seg.xlsx")
        process_excel(input_path, output_path)

    print("全部分词处理完成！")
