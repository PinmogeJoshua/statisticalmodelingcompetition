import os
import pandas as pd
from tqdm import tqdm

def combine_and_save_data(input_dir, output_dir):
    """
    合并所有品牌的数据并保存到文件
    Args:
        input_dir: 输入文件夹路径
        output_dir: 输出文件夹路径
    Returns:
        combined_df: 合并后的数据框
    """
    # 获取所有输入文件
    input_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.xlsx')]
    if not input_files:
        print(f"错误: 目录 {input_dir} 中没有找到xlsx文件")
        return None

    all_data = []

    # 遍历所有输入文件
    for file in tqdm(input_files, desc="处理文件"):
        brand_name = os.path.splitext(file)[0]
        input_path = os.path.join(input_dir, file)
        print(f"正在处理品牌: {brand_name}")
        
        try:
            df = pd.read_excel(input_path, engine='openpyxl')
            if df.empty:
                print(f"警告: 文件 {input_path} 为空，跳过")
                continue
            
            # 添加品牌列
            df['brand'] = brand_name
            all_data.append(df)
        except Exception as e:
            print(f"处理文件 {input_path} 时出错: {str(e)}")
            continue

    if not all_data:
        print("没有成功处理任何文件")
        return None

    # 合并所有数据
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_output_path = os.path.join(output_dir, "all_brands_sentiment.xlsx")
    combined_df.to_excel(combined_output_path, index=False)
    print(f"所有品牌数据已合并并保存到: {combined_output_path}")
    return combined_df
