# GoldenSense 部署指南

本系统支持在本地环境或云端 (如 Streamlit Cloud) 快速部署。

## 1. 本地部署

### 系统要求
- Python 3.9 或更高版本
- 网络连接 (用于实时抓取 yfinance 数据及新闻 RSS)

### 安装依赖
```bash
pip install -r requirements.txt
```

### 数据初始化
系统需要预先训练模型并缓存数据：
```bash
python3 train_stacking.py
```
*注：此过程将生成 `prediction_results.csv`, `raw_market_data.csv` 等关键文件。*

### 启动服务
```bash
python3 -m streamlit run pro_dashboard.py
```

## 2. 云端部署 (Streamlit Cloud)

1. 将代码推送至 GitHub 仓库。
2. 登录 [Streamlit Cloud](https://share.streamlit.io/)。
3. 点击 "New app"，选择对应的仓库、分支及主文件 `pro_dashboard.py`。
4. 在 "Secrets" 中配置必要的环境变量（如有）。
5. 点击 "Deploy" 即可完成。

## 3. 核心文件说明
- `stacking_model.py`: 模型架构与 Stacking 逻辑。
- `data_loader.py`: 数据采集与 NLP 情感处理。
- `feature_engineer.py`: 黄金市场专属特征工程。
- `pro_dashboard.py`: Streamlit 交互式前端。
- `train_stacking.py`: 模型训练与 A/B 测试脚本。
