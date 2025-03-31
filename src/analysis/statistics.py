import os
import pandas as pd
from snownlp import SnowNLP
from tqdm import tqdm
import glob
import chardet
import traceback
from openpyxl import load_workbook

def detect_file_encoding(file_path):
    """更健壮的文件编码检测"""
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read(10000)  # 只读取前10000字节来判断编码
            result = chardet.detect(raw_data)
            return result['encoding'] if result['confidence'] > 0.7 else 'utf-8'
    except:
        return 'utf-8'

def load_custom_sentiment_dicts(sentiment_dir):
    """增强的情感词典加载，添加格式验证"""
    positive_words = set()
    negative_words = set()
    
    if not os.path.exists(sentiment_dir):
        raise FileNotFoundError(f"情感词典目录不存在: {sentiment_dir}")

    for file_name in os.listdir(sentiment_dir):
        file_path = os.path.join(sentiment_dir, file_name)
        if file_name.endswith(".txt") and os.path.isfile(file_path):
            print(f"正在加载文件: {file_path}")
            try:
                encoding = detect_file_encoding(file_path)
                with open(file_path, "r", encoding=encoding, errors="ignore") as f:
                    lines = [line.strip() for line in f if line.strip()]
                    
                    # 验证词典内容
                    if not lines:
                        print(f"警告: 文件 {file_name} 为空")
                        continue
                        
                    if "正面" in file_name:
                        positive_words.update(lines)
                    elif "负面" in file_name:
                        negative_words.update(lines)
            except Exception as e:
                print(f"加载文件 {file_name} 时出错: {e}\n{traceback.format_exc()}")

    print(f"正面词语数量: {len(positive_words)}, 负面词语数量: {len(negative_words)}")
    return positive_words, negative_words

def safe_read_file(input_file):
    """更安全的文件读取方法，处理各种格式问题"""
    try:
        if input_file.endswith('.xlsx'):
            # 先检查Excel文件是否有效
            try:
                wb = load_workbook(input_file)
                wb.close()
            except Exception as e:
                print(f"Excel文件 {input_file} 可能损坏: {e}")
                return None
                
            # 尝试不同的读取方式
            for engine in ['openpyxl', 'xlrd']:
                try:
                    return pd.read_excel(input_file, engine=engine)
                except:
                    continue
            return None
        else:
            # 对于CSV，尝试多种编码和分隔符
            encodings = ['utf-8', 'gbk', 'gb18030', None]
            for encoding in encodings:
                try:
                    return pd.read_csv(input_file, encoding=encoding, on_bad_lines='skip', sep=None)
                except:
                    continue
            return None
    except Exception as e:
        print(f"读取文件 {input_file} 时出错: {e}")
        return None

def analyze_sentiment_with_custom_dict(text, positive_words, negative_words):
    """增强的情感分析函数，添加更多边界条件处理"""
    try:
        if not isinstance(text, str) or not text.strip():
            return 0.0
            
        # 添加长度检查
        if len(text) > 10000:  # 处理超长文本
            text = text[:10000]
            
        try:
            s = SnowNLP(text)
            words = s.words
        except:
            # 如果分词失败，使用简单分词
            words = text.split()
            
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        total_count = positive_count + negative_count

        # 更安全的比例计算
        if total_count == 0:
            # 无情感词时，使用SnowNLP的默认情感分析
            try:
                return SnowNLP(text).sentiments - 0.5  # 转换为-0.5到0.5范围
            except:
                return 0.0
                
        # 标准化到-1到1范围
        score = (positive_count - negative_count) / total_count
        return max(-1.0, min(1.0, score))  # 确保在范围内
    except Exception as e:
        print(f"情感分析失败: {e}\n文本: {text[:100]}...")
        return 0.0

def process_file(input_file, output_dir, text_column, positive_words, negative_words):
    """增强的文件处理函数，添加更多验证和错误处理"""
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
