"""增强版情感分析调度器"""
import os
import sys
import pandas as pd
import jieba
from tqdm import tqdm

# 添加项目根目录到 sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from src.utils.sentiment_utils import load_custom_sentiment_dicts

from src.utils.advanced_sentiment_utils import (
    expand_emotion_dict, 
    categorize_sentiment
)

from src.analysis.data_combiner import combine_and_save_data

from src.utils.visualization import (
    visualize_sentiment_distribution,
    generate_brand_wordclouds,
    plot_dimension_comparison
)

from src.analysis.construct_mood_dict import calculate_sentiment_score

from src.analysis.sentiment_analysis import (
    DIMENSIONS,
    analyze_review_dimensions,
    extract_topics_from_reviews
)

from src.analysis.statistics import calculate_emotion_proportions

class EnhancedSentimentAnalyzer:
    """增强版情感分析器类"""
    
    def __init__(self, input_dir, output_dir, results_dir=None):
        """
        初始化情感分析器
        
        Args:
            input_dir: 输入文件夹路径
            output_dir: 输出文件夹路径
            results_dir: 结果分析文件夹路径（如可视化图表等）
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.results_dir = results_dir or os.path.join(output_dir, "analysis")
        
        # 初始化情感词典
        self.positive_words = {"好": 10, "棒": 8, "满意": 7}
        self.negative_words = {"差": -10, "糟糕": -8, "失望": -7}
        
        # 尝试加载自定义词典
        try:
            sentiment_dir = os.path.join(project_root, "data/sentiment_dict")
            if os.path.exists(sentiment_dir):
                custom_pos, custom_neg = load_custom_sentiment_dicts(sentiment_dir)
                if custom_pos:
                    self.positive_words.update({word: 8 for word in custom_pos})
                if custom_neg:
                    self.negative_words.update({word: -8 for word in custom_neg})
        except Exception as e:
            print(f"加载自定义词典时出错: {str(e)}")
    
    def process_file(self, input_path, output_path):
        """
        处理单个文件
        
        Args:
            input_path: 输入文件路径
            output_path: 输出文件路径
        """
        try:
            df = pd.read_excel(input_path, engine='openpyxl')
            if df.empty:
                print(f"警告: 文件 {input_path} 为空")
                return None
                
            # 确认评论列存在
            comment_col = None
            for col in ['comment', '评论', '内容', 'Comment']:
                if col in df.columns:
                    comment_col = col
                    break
                    
            if not comment_col:
                print(f"警告: 文件 {input_path} 中未找到评论列")
                return None
            
            # 从评论中构建语料库
            corpus = []
            for comment in df[comment_col].dropna().astype(str):
                words = list(jieba.cut(comment))
                corpus.append(words)
            
            # 扩展情感词典
            expanded_positive_words = expand_emotion_dict(corpus, self.positive_words)
            expanded_negative_words = expand_emotion_dict(corpus, self.negative_words)
            
            # 基础情感分析
            df['sentiment_score'] = df[comment_col].apply(
                lambda x: calculate_sentiment_score(
                    str(x), expanded_positive_words, expanded_negative_words
                ) if isinstance(x, str) else None
            )
            
            # 添加情感分类
            df['sentiment_category'] = df['sentiment_score'].apply(
                lambda x: categorize_sentiment(x) if pd.notnull(x) else None
            )
            
            # 多维度情感分析
            df = analyze_review_dimensions(
                df, DIMENSIONS, expanded_positive_words, expanded_negative_words
            )
            
            # 提取主题
            try:
                topics, topic_sentiments, df = extract_topics_from_reviews(df)
            except Exception as e:
                print(f"主题提取失败: {str(e)}")
                topics, topic_sentiments = [], {}
            
            # 保存结果
            df.to_excel(output_path, index=False)
            
            # 保存主题信息
            if topics:
                topic_df = pd.DataFrame([
                    {'topic_id': t['topic_id'], 
                     'words': ','.join(t['words']),
                     'sentiment_score': topic_sentiments.get(t['topic_id'], None)}
                    for t in topics
                ])
                topic_path = os.path.join(
                    self.results_dir, 
                    f"{os.path.splitext(os.path.basename(input_path))[0]}_topics.xlsx"
                )
                os.makedirs(os.path.dirname(topic_path), exist_ok=True)
                topic_df.to_excel(topic_path, index=False)
            
            return df
            
        except Exception as e:
            print(f"处理文件 {input_path} 时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_analysis(self):
        """运行情感分析"""
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 获取所有输入文件
        input_files = [f for f in os.listdir(self.input_dir) if f.lower().endswith('.xlsx')]
        if not input_files:
            print(f"错误: 目录 {self.input_dir} 中没有找到xlsx文件")
            return
        
        # 处理所有文件
        all_data = []
        stats_data = []
        
        for file in tqdm(input_files, desc="处理文件"):
            # 提取品牌名称
            brand_name = os.path.splitext(file)[0]
            input_path = os.path.join(self.input_dir, file)
            output_path = os.path.join(self.output_dir, f"{brand_name}_sentiment.xlsx")
            
            print(f"正在处理品牌: {brand_name}")
            df = self.process_file(input_path, output_path)
            
            if df is not None:
                # 添加品牌列
                df['brand'] = brand_name
                all_data.append(df)
                
                # 计算情感比例
                stats = calculate_emotion_proportions(output_path)
                if stats:
                    stats['brand'] = brand_name
                    stats_data.append(stats)
        
        if not all_data:
            print("没有成功处理任何文件")
            return
        
        # 保存统计数据
        if stats_data:
            stats_df = pd.DataFrame(stats_data)
            stats_df.to_excel(os.path.join(self.results_dir, "brand_sentiment_stats.xlsx"), index=False)
        
        # 合并所有数据并生成可视化
        combined_df = combine_and_save_data(self.output_dir, self.results_dir)
        if combined_df is not None:
            visualize_sentiment_distribution(combined_df, self.results_dir)
            generate_brand_wordclouds(combined_df, self.results_dir)
            plot_dimension_comparison(combined_df, self.results_dir)
        
        print("情感分析完成！结果已保存到:", self.output_dir)
        print("分析报告已保存到:", self.results_dir)
        