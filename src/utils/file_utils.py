"""文件工具相关函数"""
import chardet

def detect_file_encoding(file_path):
    """更健壮的文件编码检测"""
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read(10000)  # 只读取前10000字节来判断编码
            result = chardet.detect(raw_data)
            return result['encoding'] if result['confidence'] > 0.7 else 'utf-8'
    except:
        return 'utf-8'