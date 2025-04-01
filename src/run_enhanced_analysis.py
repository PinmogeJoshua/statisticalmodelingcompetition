"""运行增强版情感分析"""
import os
import sys
import argparse
from src.analysis.enhanced_sentiment_analyzer import EnhancedSentimentAnalyzer
from src.utils.visualization import plot_dimension_comparison
from src.analysis.data_combiner import combine_and_save_data

# 添加项目根目录到 sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="运行增强版情感分析")
    
    parser.add_argument(
        "--input", 
        default=os.path.join(project_root, "data/segment_results"),
        help="输入文件夹路径，包含分词后的Excel文件"
    )
    
    parser.add_argument(
        "--output", 
        default=os.path.join(project_root, "data/results/advanced_analysis/mood_results"),
        help="输出文件夹路径，保存情感分析结果"
    )
    
    parser.add_argument(
        "--results", 
        default=os.path.join(project_root, "data/results/advanced_analysis/visualization"),
        help="分析结果文件夹路径，保存可视化图表和统计数据"
    )
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    print("增强版情感分析开始...")
    print(f"输入文件夹: {args.input}")
    print(f"输出文件夹: {args.output}")
    print(f"结果文件夹: {args.results}")
    
    # 运行情感分析
    analyzer = EnhancedSentimentAnalyzer(
        input_dir=args.input,
        output_dir=args.output,
        results_dir=args.results
    )
    analyzer.run_analysis()
    
    # 合并所有数据
    print("开始合并所有品牌的数据...")
    combined_df = combine_and_save_data(args.output, args.results)
    if combined_df is not None:
        print("数据合并完成，开始生成可视化图表...")
        # 调用 plot_dimension_comparison
        plot_dimension_comparison(
            df=combined_df,
            output_dir=args.results,
            brand_column="brand"
        )
        print("可视化图表生成完成！")
    else:
        print("错误: 无法生成 combined_df，无法绘制图表")

if __name__ == "__main__":
    main()
    