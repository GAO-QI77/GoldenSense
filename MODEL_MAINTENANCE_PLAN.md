# AI Model Maintenance & Retraining Strategy

**Project**: GoldenSense  
**Version**: 1.0  
**Date**: 2026-03-16

---

## 1. Port Conflict Resolution & Prevention

### Diagnosis
Port conflicts occur when the server process (`uvicorn`) is not terminated properly or when multiple instances are launched.
*   **Check Command**: `lsof -i :8000` (MacOS/Linux) or `netstat -ano | findstr :8000` (Windows).
*   **Kill Command**: `kill -9 <PID>` (MacOS/Linux) or `taskkill /PID <PID> /F` (Windows).

### Solution Implemented
The `server.py` startup logic has been modified to automatically detect port availability:
1.  Check port 8000.
2.  If occupied, increment port number (8001, 8002...) until a free port is found.
3.  Log the selected port to stdout.

### Best Practices
*   Use a process manager like `supervisord` or `systemd` in production.
*   Implement a `shutdown` hook in FastAPI to close connections gracefully.

---

## 2. Model Retraining Strategy

### 2.1 Business Context Analysis
*   **Asset**: Gold (XAU/USD).
*   **Market Regime**: High volatility, influenced by geopolitical events and macro data (CPI, NFP).
*   **Data Velocity**: 
    *   Price: Tick-level (aggregated to daily/hourly).
    *   News: Real-time.
*   **Concept Drift**: Significant. A model trained on 2020 (COVID) data may fail in 2024 (Rate Cuts).

### 2.2 Retraining Frequency
Based on the decay curve of financial time-series models:

| Frequency | Trigger | Scope | Resource Cost |
| :--- | :--- | :--- | :--- |
| **Weekly (Recommended)** | Every Sunday | Full Retrain (Rolling Window 2Y) | Medium (10-20 mins) |
| **Daily (Incremental)** | New data available | Online Learning / Fine-tuning | Low (< 2 mins) |
| **Event-Driven** | Major Regime Change (e.g., War, Rate Hike) | Full Retrain + Hyperparam Search | High (1-2 hours) |

**Decision**: Adopt a **Weekly Full Retrain** strategy with a rolling window of 2 years to capture recent market dynamics while discarding obsolete patterns.

### 2.3 Monitoring & Alerts

Define Key Performance Indicators (KPIs) to trigger emergency retraining:

*   **Accuracy (Directional)**: < 50% for 3 consecutive days.
*   **MAE (Price)**: > $50 deviation.
*   **Data Drift**: Input distribution shift > 20% (KS-Test).

### 2.4 Rollback Strategy

Always keep the previous best model.

1.  **Backup**: Before saving new model, rename `model_checkpoints` to `model_checkpoints_backup`.
2.  **Validation**: New model must beat the old model on the last 30 days of hold-out data.
3.  **Rollback**: If new model performs worse in production (24h grace period), restore from backup.

### 2.5 Automation Pipeline

```bash
# Weekly Cron Job (Sunday 00:00)
0 0 * * 0 /usr/bin/python3 /path/to/train_stacking.py --mode=production >> /var/log/goldensense/retrain.log 2>&1
```

## 3. Resource Cost Analysis

*   **Training**: ~15 mins on CPU (M1/M2). Cost negligible.
*   **Inference**: < 50ms.
*   **Storage**: < 500MB per checkpoint version.

**Conclusion**: The Weekly Retrain strategy offers the best balance between model freshness and operational stability.
