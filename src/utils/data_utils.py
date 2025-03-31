"""数据读取相关工具"""
import pandas as pd
from openpyxl import load_workbook

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
