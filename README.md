<div align="center">

# 🎯 Meta-AC

### 面向学术评审的智能 Area Chair 系统

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

*利用多智能体系统和机器学习技术，自动化学术论文评审决策过程*

[功能特性](#-功能特性) •
[快速开始](#-快速开始) •
[使用指南](#-使用指南) •
[项目结构](#-项目结构) •
[贡献指南](#-贡献指南)

</div>

---

## 📖 项目简介

Meta-AC 是一个智能化的学术评审辅助系统，专门设计用于处理和分析 ICLR 等顶级会议的论文评审数据。系统通过以下方式提供决策支持：

- 🔍 **数据解析**：自动解析 OpenReview 评审数据
- 🤖 **多智能体系统**：集成 Bayesian、Argument、Domain 等多个智能代理
- 📊 **机器学习预测**：使用 MLP 神经网络预测论文接受概率
- 📈 **可视化分析**：生成直观的数据可视化图表
- 🎨 **交互式界面**：提供 Streamlit 仪表盘进行实时探索

---

## ✨ 功能特性

### 🔥 核心功能

| 功能模块 | 描述 |
|---------|------|
| 📝 **数据预处理** | 解析原始 JSON 数据，智能抽样，保持类别平衡 |
| 🧠 **Bayesian Agent** | 贝叶斯校准评分，处理评审者信心权重 |
| 💬 **Argument Agent** | LLM 驱动的反驳质量评估（支持 DeepSeek API） |
| 🎓 **Domain Agent** | 领域特定的论文新颖度分析 |
| 🤖 **MLP 分类器** | 多层感知机神经网络进行接受概率预测 |
| 📊 **数据可视化** | 生成评分分布和预测影响力图表 |
| 🖥️ **交互式仪表盘** | Streamlit 支持的 Web 界面 |

### 🎨 技术栈

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-121212?style=for-the-badge&logo=chainlink&logoColor=white)

</div>

---

## 🚀 快速开始

### 📋 环境要求

- **Python**: 3.10 或更高版本
- **系统**: Linux / macOS / Windows

### 📦 安装依赖

```bash
# 克隆仓库
git clone https://github.com/GarvinRay/Data-analysis-project.git
cd Data-analysis-project

# 安装核心依赖
pip install pandas numpy tqdm requests sentence-transformers scikit-learn

# 可选：安装可视化和 Web 界面依赖
pip install streamlit plotly

# 可选：安装 LangChain 和 LLM 相关依赖
pip install langchain langchain_openai langchain_community duckduckgo-search joblib
```

### 🔑 API 配置（可选）

如需使用 LLM 功能，配置 DeepSeek API：

```bash
export DEEPSEEK_API_KEY="your_api_key_here"
```

> 💡 **提示**：未设置 API 密钥时，系统将自动使用 mock 逻辑运行

---

## 📚 使用指南

### 🔄 完整工作流程

#### 1️⃣ 数据预处理与抽样

```bash
python process_data.py --total-samples 50
```

**输出文件：**
- `data/processed/meta_ac_dataset_sampled.json` - 结构化的论文记录
- `data/processed/meta_ac_stats_sampled.csv` - 统计特征数据

**功能说明：**
- ✅ 解析 OpenReview JSON 数据（Oral/Spotlight/Poster/Reject）
- ✅ 智能抽样，维持 Accept/Reject 1:1 平衡
- ✅ 保持 Accept 内部类别的原始比例

---

#### 2️⃣ 特征提取与模型训练

```bash
python main.py
```

**处理流程：**
1. 📖 读取采样数据
2. 🔍 提取多维度特征：
   - Bayesian 校准评分
   - 评分方差
   - 评审数量
   - LLM 反驳得分
3. 🎓 训练 MLP 分类器
4. 📊 评估测试集性能
5. 💾 生成预测结果文件

**输出文件：**
- `data/outputs/final_predictions.csv` - 完整预测结果
- `data/outputs/meta_ac_model.pkl` - 训练好的模型

---

#### 3️⃣ 数据可视化（可选）

**生成静态图表：**
```bash
python plot_analysis.py
```
输出：`meta_ac_impact.png` - 原始评分 vs Meta-AC 预测概率对比图

**启动交互式仪表盘：**
```bash
streamlit run app.py
```
在浏览器中打开 `http://localhost:8501` 进行交互式数据探索

---

#### 4️⃣ 高级功能（可选）

**LangChain ReAct Agent：**
```bash
python agent_graph.py
```
展示完整的 "Thought/Action/Observation" 推理日志

**逻辑回归训练：**
```bash
python train_model.py
```
训练逻辑回归模型并分析特征权重

---

## 📁 项目结构

```
Data-analysis-project/
│
├── 📄 main.py                    # 主训练和评估脚本
├── 📄 process_data.py            # 数据预处理和抽样
├── 📄 agents.py                  # 多智能体实现（Bayesian/Argument/Domain）
├── 📄 plot_analysis.py           # 可视化图表生成
├── 📄 app.py                     # Streamlit 仪表盘（如存在）
├── 📄 train_model.py             # 逻辑回归训练（如存在）
├── 📄 agent_graph.py             # LangChain Agent（如存在）
│
├── 📂 meta_ac/                   # 核心模块（建议结构）
│   ├── config.py                 # 配置管理（路径/参数）
│   ├── models.py                 # 数据模型定义
│   └── __init__.py
│
└── 📂 data/                      # 数据目录（建议结构）
    ├── 📂 raw/                   # 原始 OpenReview JSON 文件
    │   ├── openreview_oral_results.json
    │   ├── openreview_spotlight_results.json
    │   ├── openreview_poster_results.json
    │   └── openreview_reject_results.json
    │
    ├── 📂 processed/             # 处理后的数据
    │   ├── meta_ac_dataset_sampled.json
    │   └── meta_ac_stats_sampled.csv
    │
    └── 📂 outputs/               # 输出结果
        ├── final_predictions.csv
        ├── meta_ac_impact.png
        └── meta_ac_model.pkl
```

---

## 📊 数据格式说明

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

## ⚠️ 注意事项

> 📌 **重要提示**

- 🔐 **API 密钥**：未设置 `DEEPSEEK_API_KEY` 时，LLM 相关特征将使用 mock 数据
- ⚖️ **数据平衡**：采样比例由 `--total-samples` 参数控制，自动维持 Accept/Reject 1:1 平衡
- 🏗️ **鲁棒性**：`main.py` 会自动跳过异常样本，确保流程完整运行
- 📊 **比例保持**：Accept 类别内部会按原始 Oral/Spotlight/Poster 比例进行抽样

---

## 🤝 贡献指南

我们欢迎各种形式的贡献！无论是新功能、bug 修复还是文档改进。

### 贡献流程

1. 🍴 Fork 本仓库
2. 🔧 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 💾 提交变更 (`git commit -m 'Add some AmazingFeature'`)
4. 📤 推送到分支 (`git push origin feature/AmazingFeature`)
5. 🎉 提交 Pull Request

### 代码规范

- 遵循 PEP 8 编码规范
- 添加适当的类型注解
- 编写清晰的文档字符串
- 确保所有测试通过

---

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

---

## 📮 联系方式

- **项目链接**: [https://github.com/GarvinRay/Data-analysis-project](https://github.com/GarvinRay/Data-analysis-project)
- **问题反馈**: [提交 Issue](https://github.com/GarvinRay/Data-analysis-project/issues)

---

<div align="center">

### ⭐ 如果这个项目对你有帮助，请给我们一个 Star！

**Made with ❤️ by the Meta-AC Team**

</div>
