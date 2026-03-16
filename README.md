# GoldenSense: AI-Driven Gold Market Intelligence & Risk Control System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)

> **GoldenSense** is a professional-grade AI Agent system designed for real-time gold market (XAU/USD) analysis and risk management. It integrates deep learning ensemble models with real-time causal analysis of global macro-events to provide high-precision trend predictions and actionable market intelligence.

---

## 🏗 Technical Architecture

GoldenSense employs a **Multi-Horizon Stacking Ensemble** architecture, combining the strengths of sequential deep learning and gradient-boosted decision trees.

### 1. Model Engine (Stacking Ensemble)
*   **Sequential Layer**:
    *   **Transformer**: Leverages Multi-Head Self-Attention to capture long-range dependencies in global macro signals.
    *   **GRU (Gated Recurrent Unit)**: Optimized RNN architecture for short-term price momentum and volatility regimes.
*   **Tabular Layer**:
    *   **XGBoost & Random Forest**: Extracts non-linear relationships from structured market data (Real Yields, VIX, USD Index).
*   **Meta-Learner**: A non-linear fusion layer that dynamically weights sub-model outputs based on 7-day rolling performance.

### 2. Causal Data Pipeline
*   **Adaptive Normalization**: Implements rolling Z-Score mapping to handle heteroscedasticity in financial time-series.
*   **Causal NLP Engine**: Quantifies the impact of global events (Inflation, Policy, Geopolitics) using weighted causality scoring.

---

## 🚀 Core Features

- **Real-Time Inference**: Sub-100ms prediction latency using optimized PyTorch and XGBoost backends.
- **Multi-Horizon Forecasts**: Synchronized predictions for T+1 (24h), T+7 (Weekly), and T+30 (Monthly) horizons.
- **Intelligent Analysis Brief**: Auto-generated deep analysis reports combining AI predictions with current macro-narratives.
- **Robust Data Handling**: Exponential backoff retry mechanisms for real-time `yfinance` data fetching.
- **Interactive Dashboard**: Modern, responsive UI with real-time SSE (Server-Sent Events) price updates.

---

## 📂 Project Structure

```text
.
├── live_platform/          # FastAPI Live Dashboard
│   ├── static/             # Frontend (HTML5, Tailwind, Chart.js)
│   ├── server.py           # Backend SSE & Inference Server
│   └── Dockerfile          # Production deployment
├── ratelmind/              # Core Logic & Utilities
│   └── services/           # Token optimization & AI Services
├── stacking_model.py       # Multi-model Ensemble Architecture
├── feature_engineer.py     # Adaptive Normalization & Feature Construction
├── data_loader.py          # Market & News Data Pipeline
├── train_stacking.py       # Production Training & Backtesting Script
└── requirements.txt        # Dependency Specification
```

---

## 🛠 Installation & Deployment

### Prerequisites
- Python 3.9 or higher
- Git

### 1. Clone & Install
```bash
git clone https://github.com/GAO-QI77/GoldenSense.git
cd GoldenSense
pip install -r requirements.txt
```

### 2. Model Initialization
Train the production models and generate the initial feature configuration:
```bash
python3 train_stacking.py
```

### 3. Launch the Live Platform
Start the FastAPI server (supports auto-port detection):
```bash
python3 -m uvicorn live_platform.server:app --host 0.0.0.0 --port 8000 --reload
```
Access the dashboard at `http://localhost:8000`.

---

## 📊 Performance Indicators

| Metric | Target | Achieved (Backtest) |
| :--- | :--- | :--- |
| **T+1 Directional Accuracy** | >52% | **55.4%** |
| **Inference Latency** | <200ms | **72ms** |
| **System Uptime** | 99.9% | **99.95%** |
| **Token Efficiency** | -80% | **-85% (Optimized)** |

---

## 🛡 Security & Compliance

*   **API Security**: Implements `X-API-Key` header verification for all data endpoints.
*   **Data Integrity**: Checksum verification for model checkpoints.
*   **Disclaimer**: This software is for educational and research purposes only. It does not constitute financial advice. Trading gold involves significant risk.

---

## 🤝 Contribution

We welcome contributions from the community. Please follow our [Contribution Guidelines](CONTRIBUTING.md) and ensure all PRs include corresponding unit tests.

**Maintainer**: GAO-QI (GitHub: @GAO-QI77)  
**License**: MIT License
