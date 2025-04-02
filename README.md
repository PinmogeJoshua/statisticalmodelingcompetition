# StatModel: 增强版情感分析与数据可视化

## 项目简介
StatModel 是一个用于情感分析和数据可视化的 Python 项目，支持多维度情感分析、情感词典扩展、主题提取以及数据可视化。项目特别适用于中文文本的情感分析，提供了增强的分析器和丰富的可视化功能。

---

## 功能特性
- **情感分析**: 基于自定义情感词典和扩展词典的情感分析。
- **多维度分析**: 针对不同维度（如安全性、舒适性等）进行情感得分计算。
- **主题提取**: 使用 LDA 模型从评论中提取主题。
- **数据可视化**: 提供情感分布图、词云图、热图等多种可视化图表。
- **批量处理**: 支持批量处理 Excel 和 CSV 文件。

---

## 项目结构
StatModel/
├── src/
│   ├── analysis/                # 分析模块
│   │   ├── construct_mood_dict.py
│   │   ├── data_combiner.py
│   │   ├── enhanced_sentiment_analyzer.py
│   │   ├── run_sentiment_analysis.py
│   │   ├── sentiment_analysis.py
│   │   ├── statistics.py
│   ├── text_processing/         # 文本处理模块
│   │   ├── comment_cut.py
│   ├── utils/                   # 工具模块
│   │   ├── advanced_sentiment_utils.py
│   │   ├── data_utils.py
│   │   ├── file_utils.py
│   │   ├── sentiment_utils.py
│   │   ├── visualization.py
│   ├── run_enhanced_analysis.py # 主运行脚本
├── requirements.txt             # 项目依赖
└── README.md                    # 项目说明

---

## 安装依赖
在运行项目之前，请确保安装所有依赖包。可以通过以下命令安装：

```bash
pip install -r [requirements.txt](http://_vscodecontentref_/5)
```

---

## 使用方法
1. 数据准备
将需要分析的 Excel 或 CSV 文件放入 data/segment_results 文件夹中。文件需包含评论列（如 comment 或 内容）。

2. 运行情感分析
运行以下命令启动增强版情感分析：
```bash
pip [run_enhanced_analysis.py](http://_vscodecontentref_/6) --input data/segment_results --output data/results --results data/visualization
```
参数说明：
--input: 输入文件夹路径，包含待分析的文件。
--output: 输出文件夹路径，保存分析结果。
--results: 结果文件夹路径，保存可视化图表和统计数据。

3. 查看结果
分析结果: 在 data/results 文件夹中查看情感分析后的文件。
可视化图表: 在 data/visualization 文件夹中查看生成的图表（如词云图、热图等）。

---

## 依赖包
请查看requirements.txt

---

## 注意事项
确保输入文件的编码格式正确（推荐 UTF-8）。

如果需要支持中文词云，请确保系统中安装了中文字体（如 SimHei 字体）。

---

## 开发者
作者：Pinmoge

邮箱：stellewang0417@qq.com

---

## 许可证
本项目基于 MIT 许可证开源
