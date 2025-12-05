## Meta-AC 项目说明

面向 ICLR/OpenReview 评审数据的分析pipeline，提供数据预处理、特征抽取、多模型训练与可视化。目标是构建一个可复现、可审计的评审辅助流程，而非强调模型“智能”。

---

## 功能概览
- 数据预处理：解析多个 OpenReview JSON（Oral/Spotlight/Poster/Reject），按类别平衡抽样。
- 特征工程：BayesianAgent 计算校准分，ArgumentAgent 评估反驳质量（可调用 DeepSeek API，未配置时使用 mock），DomainAgent 计算新颖度/密度。
- 模型训练：默认 MLP 分类器，支持网格搜索；可选 TabNet（需安装依赖）。
- 结果产出：生成预测 CSV、模型权重/指标、可视化图表；Streamlit 仪表盘用于交互浏览。
- 复现记录：保存 run_metrics.json、misclassified.csv、feature_weights.csv 等，便于实验留痕。

---

## 快速开始

### 环境
- Python 3.10+

### 安装依赖
```bash
pip install pandas numpy tqdm sentence-transformers scikit-learn joblib streamlit seaborn matplotlib
```

### API 密钥（可选，ArgumentAgent 调用 DeepSeek）
```bash
export DEEPSEEK_API_KEY="your_key"
# 未配置则使用 mock 评分
```

---

## 使用流程

1) 数据预处理与抽样  
```bash
python process_data.py --total-samples 1000
```
输出：
- `data/processed/meta_ac_dataset_sampled.json`
- `data/processed/meta_ac_stats_sampled.csv`

2) 特征提取 + 模型训练  
```bash
# 默认 MLP
python main.py
# 启用网格搜索
python main.py --grid-search
# 尝试 TabNet（需依赖）
python main.py --model-type TABNET
```
输出：
- `data/outputs/final_predictions.csv`
- `data/outputs/meta_ac_model.pkl`
- `data/outputs/run_metrics.json`
- `data/outputs/misclassified.csv`
- `data/outputs/feature_weights.csv`

3) 可视化  
```bash
python plot_analysis.py
```
输出图表位于 `data/outputs/`。

4) 交互式仪表盘  
```bash
streamlit run app.py
```

---

## 项目结构（简要）
```
data-analysis-project/
├── main.py                 # 训练与评估
├── process_data.py         # 解析与抽样
├── agents.py               # Bayesian/Argument/Domain 代理
├── plot_analysis.py        # 可视化
├── app.py                  # Streamlit 仪表盘
├── meta_ac/
│   ├── config.py           # 路径配置
│   ├── models.py           # 数据模型
│   └── __init__.py
└── data/
    ├── raw/                # 原始 OpenReview JSON
    ├── processed/          # 处理后数据
    │   ├── meta_ac_dataset_sampled.json
    │   └── meta_ac_stats_sampled.csv
    │
    └── outputs/               # 输出结果
        ├── final_predictions.csv
        ├── meta_ac_impact.png
        └── meta_ac_model.pkl
```

---

## 数据格式说明

### 输入数据

| 文件 | 格式 | 说明 |
|------|------|------|
| `openreview_*_results.json` | JSON | OpenReview 原始评审数据 |

### 中间数据

| 文件 | 格式 | 内容 |
|------|------|------|
| `meta_ac_dataset_sampled.json` | JSON | `PaperRecord.to_dict()` 数组，包含评审/反驳对齐文本 |
| `meta_ac_stats_sampled.csv` | CSV | 量化特征：avg_rating, rating_variance, confidence_weighted_avg, num_reviews 等 |

### 输出数据

| 文件 | 格式 | 内容 |
|------|------|------|
| `final_predictions.csv` | CSV | paper_id, 模型预测概率, 原始标签 |

---

## 注意事项

- API 密钥：未设置 `DEEPSEEK_API_KEY` 时，ArgumentAgent 使用 mock 得分。
- 数据平衡：`--total-samples` 控制采样量，Accept/Reject 维持 1:1。
- 鲁棒性：`main.py` 会跳过异常样本，流程持续运行。
- 比例保持：Accept 内部保持 Oral/Spotlight/Poster 原始比例。

