# GoldenSense 🏆 

> **基于 Stacking 集成学习的黄金价格多周期智能预测系统**

GoldenSense 是一款专为金融实战设计的 AI 黄金价格预测平台。它结合了深度学习（Transformer, GRU）与经典机器学习（XGBoost, 随机森林）的优势，通过 Stacking 集成架构实现高稳健性的市场预测。

## ✨ 核心特性

- **Stacking 混合架构**：深度融合 Transformer (捕捉情绪)、GRU (时序记忆) 与树模型 (特征提取)，由 MLP 元学习器实现非线性权重分配。
- **多周期预测 (Multi-Horizon)**：同步提供 T+1 (明日)、T+7 (中期) 及 T+30 (月度) 的价格趋势与涨跌概率。
- **深度因果分析管道**：实时量化路透、彭博等权威资讯，提取通胀、利率、避险及汇率四大核心驱动因子。
- **智能深度简报**：AI 自动生成 300 字市场深度分析，涵盖影响机制评估、置信度评分及投资建议。
- **专业级交互仪表板**：基于 Streamlit 开发，具备毛玻璃视觉风格，支持实时行情、技术指标及历史回测验证。

## 🛠️ 技术栈

- **模型引擎**: PyTorch (Transformer, GRU), XGBoost, Scikit-learn
- **特征工程**: Pandas, Numpy, TA-Lib (模拟)
- **数据来源**: yfinance (市场数据), RSS Feeds (实时新闻)
- **可视化/Web**: Streamlit, Plotly, Custom CSS
- **NLP**: 语义因果映射 (Causal Lexicon)

## 🚀 快速启动

### 1. 环境安装
```bash
pip install -r requirements.txt
```

### 2. 模型训练与数据生成
```bash
python3 train_stacking.py
```

### 3. 启动交互仪表板
```bash
python3 -m streamlit run pro_dashboard.py
```

## 📈 系统展示 (截图占位)

*(此处可放置 dashboard.png)*

## 📄 开源协议

本项目采用 MIT 协议开源。

---
**Team**: Golden is everything | **Hackathon Submission**
