```markdown
# Meta-AC：面向学术评审的智能 Area Chair 系统

## 项目简介
Meta-AC 处理 ICLR 评审数据，解析与抽样，利用多智能体（Bayesian / Argument / Domain 等）与机器学习模型对论文做出接受概率预测，支持可视化与训练/评估。

## 目录结构（推荐）
- `meta_ac/`  
  - `config.py`：集中管理数据/输出路径。  
  - `models.py`：数据模型（`PaperRecord`, `ReviewRebuttalPair` 等）。  
  - `__init__.py`
- `process_data.py`：解析原始 OpenReview JSON，抽样，生成处理后的数据。  
- `main.py`：调用 agents 抽取特征，训练/评估 MLP，并输出预测。  
- `agents.py`：Bayesian/Argument/Domain 等代理逻辑（LLM/校准/新颖度）。  
- `plot_analysis.py`：生成 Raw Score vs Meta-AC Probability 的图。  
- `app.py`：Streamlit 仪表盘。  
- `train_model.py`：逻辑回归训练（可选）。  
- `agent_graph.py`：LangChain ReAct Agent（可选）。  
- `data/`（建议）  
  - `raw/`：原始 JSON（openreview_oral/spotlight/poster/reject 等）。  
  - `processed/`：`meta_ac_dataset_sampled.json`, `meta_ac_stats_sampled.csv`。  
  - `outputs/`：`final_predictions.csv`, `meta_ac_impact.png`, `meta_ac_model.pkl` 等。

## 依赖
Python 3.10+，主要依赖：`pandas`, `numpy`, `tqdm`, `requests`, `sentence-transformers`, `scikit-learn`, `streamlit`, `plotly`, `langchain`, `langchain_openai`, `langchain_community`, `duckduckgo-search`, `joblib` 等。

如需调用 DeepSeek API，请设置：
```bash
export DEEPSEEK_API_KEY="your_api_key_here"
```
未设置时，`ArgumentAgent` 会回退到 mock 逻辑。

## 典型流程
1) **数据预处理与抽样**  
```bash
python process_data.py --total-samples 50
```
输出到 `data/processed/meta_ac_dataset_sampled.json` 和 `meta_ac_stats_sampled.csv`。

2) **特征提取 + MLP 训练/评估**  
```bash
python main.py
```
读取采样数据，提取特征（Bayesian 校准分、方差、评审数、LLM 反驳得分），训练 MLP，输出测试集指标，并将全量预测写入 `data/outputs/final_predictions.csv`。

3) **可视化/仪表盘（可选）**  
- `python plot_analysis.py` 生成 `meta_ac_impact.png`。  
- `streamlit run app.py` 交互式查看。

4) **其他（可选）**  
- `agent_graph.py`：LangChain ReAct Agent，展示“Thought/Action”日志。  
- `train_model.py`：逻辑回归训练与特征权重分析。

## 数据格式
- `meta_ac_dataset_sampled.json`：由 `PaperRecord.to_dict()` 组成的数组，包含评审/反驳对齐的文本。  
- `meta_ac_stats_sampled.csv`：量化特征（avg_rating, rating_variance, confidence_weighted_avg, num_reviews 等）。  
- `final_predictions.csv`：`paper_id`、模型概率、原始标签。

## 注意
- 若未设置 `DEEPSEEK_API_KEY`，LLM 相关特征将使用 mock。  
- 采样比例由 `process_data.py --total-samples` 控制，Accept/Reject 维持 1:1，Accept 内部按 Oral/Spotlight/Poster 原比例。  
- `main.py` 对异常样本会跳过继续，确保全量运行。
```

