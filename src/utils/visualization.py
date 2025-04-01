"""数据可视化工具"""
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import jieba

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def visualize_sentiment_distribution(df, output_dir, brand_column='brand', sentiment_column='sentiment_score'):
    """可视化各品牌的情感得分分布"""
    plt.figure(figsize=(12, 6))
    
    # 修复 Seaborn 的 FutureWarning
    sns.violinplot(x=brand_column, y=sentiment_column, data=df, palette="Set3", hue=None, legend=False)
    plt.title('各品牌情感得分分布')
    plt.xlabel('品牌')
    plt.ylabel('情感得分')
    
    # 保存图表
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sentiment_distribution.png'), dpi=300)
    plt.close()
    
    # 绘制箱线图
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=brand_column, y=sentiment_column, data=df, palette="Set2", hue=None, legend=False)
    plt.title('各品牌情感得分箱线图')
    plt.xlabel('品牌')
    plt.ylabel('情感得分')
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sentiment_boxplot.png'), dpi=300)
    plt.close()

def load_stopwords(filepath):
    """加载停用词列表"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return set(line.strip() for line in f if line.strip())
    
# 加载停用词
STOPWORDS = load_stopwords("c:\\VSCode\\StatisticalModeling\\cn_stopwords.txt")

def generate_brand_wordclouds(df, output_dir, brand_column='brand', comment_column='comment'):
    """为每个品牌生成词云图
    
    Args:
        df: 包含品牌和评论的DataFrame
        output_dir: 输出目录
        brand_column: 品牌列名
        comment_column: 评论列名
    """
    os.makedirs(output_dir, exist_ok=True)
    brands = df[brand_column].unique()
    
    for brand in brands:
        brand_comments = df[df[brand_column] == brand][comment_column].dropna().astype(str)
        if brand_comments.empty:
            continue
            
        # 拼接所有评论
        text = ' '.join(brand_comments)
        
        # 使用 jieba 分词
        words = ' '.join(jieba.cut(text))
        
        # 创建词云对象，设置停用词
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white', 
            font_path='simhei.ttf',  # 确保这个字体文件存在或修改为系统中存在的中文字体
            max_words=100,
            stopwords=STOPWORDS  # 使用加载的停用词
        )
        
        try:
            # 生成词云
            wordcloud.generate(words)
            
            # 保存词云图
            output_path = os.path.join(output_dir, f'{brand}_wordcloud.png')
            wordcloud.to_file(output_path)
            print(f"词云图已保存: {output_path}")
        except Exception as e:
            print(f"生成 {brand} 的词云图时出错: {str(e)}")

def plot_dimension_comparison(df, output_dir, brand_column='brand'):
    """绘制各品牌在不同维度上的情感得分对比"""
    os.makedirs(output_dir, exist_ok=True)
    print("图表保存路径:", output_dir)  # 打印保存路径
    
    # 打印数据框的列名，检查是否包含 'category'
    print("数据框的列名:", df.columns.tolist())

    # 找出所有维度列，排除 sentiment_category
    dimension_columns = [
        col for col in df.columns 
        if col.startswith('sentiment_') and col != 'sentiment_score' and col != 'sentiment_category'
    ]
    if not dimension_columns:
        print("警告: 未找到维度情感得分列")
        return
    
    # 计算各品牌在各维度的平均得分
    brands = df[brand_column].unique()
    result_data = []
    
    for brand in brands:
        brand_data = df[df[brand_column] == brand]
        row = {'brand': brand}
        
        for col in dimension_columns:
            dimension = col.replace('sentiment_', '')
            
            # 确保列中只有数值数据
            numeric_data = pd.to_numeric(brand_data[col], errors='coerce')
            if numeric_data.notna().any():
                row[dimension] = numeric_data.mean()
            else:
                print(f"警告: 品牌 {brand} 的维度 {dimension} 数据非数值，跳过计算")
                row[dimension] = None
            
        result_data.append(row)
    
    result_df = pd.DataFrame(result_data)
    print("计算结果数据框:\n", result_df)  # 打印计算结果数据框
    
    # 转换为绘图所需的长格式
    plot_df = pd.melt(
        result_df, 
        id_vars=['brand'], 
        value_vars=[col for col in result_df.columns if col != 'brand'],
        var_name='dimension', 
        value_name='score'
    )
    
    # 绘制雷达图
    print("正在绘制雷达图...")
    plt.figure(figsize=(14, 10))
    for brand in brands:
        brand_data = plot_df[plot_df['brand'] == brand]
        plt.plot(brand_data['dimension'], brand_data['score'], marker='o', label=brand)
    
    plt.title('各品牌在不同维度的情感得分对比')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dimension_comparison.png'), dpi=300)
    plt.close()
    
    # 绘制热图
    print("正在绘制热图...")
    pivot_df = plot_df.pivot(index='brand', columns='dimension', values='score')
    
    # 确保 pivot_df 中的数据为数值类型
    pivot_df = pivot_df.apply(pd.to_numeric, errors='coerce')  # 将非数值数据转换为 NaN
    if pivot_df.isnull().values.any():
        print("警告: pivot_df 包含 NaN 值，可能影响热图绘制")
        pivot_df = pivot_df.fillna(0)  # 用 0 填充 NaN 值
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_df, annot=True, cmap='RdYlGn', linewidths=.5)
    plt.title('各品牌在不同维度的情感得分热图')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dimension_heatmap.png'), dpi=300)
    plt.close()
    