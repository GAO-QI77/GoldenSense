# GoldenSense Model Maintenance & Retraining Strategy

- **Project**: GoldenSense
- **Version**: 1.1
- **Date**: 2026-05-12

---

## 1. Current Runtime Model

GoldenSense runs as a small service mesh rather than a single `server.py` entrypoint:

| Service | Default port | Role |
| --- | --- | --- |
| `inference_service.py` | `8010` | Loads checkpoints and returns horizon forecasts |
| `memory_service.py` | `8012` | Searches historical event memory |
| `market_snapshot_service.py` | `8014` | Provides market snapshots and indicator groups |
| `news_ingest_service.py` | `8016` | Normalizes recent news |
| `agent_gateway.py` | `8020` | Orchestrates the public agent API |

Use `zsh scripts/dev_stack.sh start` for local backend development, or
`docker compose up --build` for the full local stack. If a port is occupied,
stop the existing process with `zsh scripts/dev_stack.sh stop` or inspect the
port with `lsof -i :<port>`.

---

## 2. Asset Policy

The tracked files in `model_checkpoints/` and the small CSV/JSON research files
are lightweight demo assets used by local inference, training, tests, and
documentation examples. They are intentionally kept small enough for this public
repository.

Do not commit private datasets, customer data, provider credentials, or large
production model versions. Use one of these instead:

- GitHub Release artifacts for versioned demo bundles
- Object storage for production datasets and large checkpoints
- Git LFS only when repository-level binary versioning is truly required

---

## 3. Retraining Strategy

### 3.1 Business Context

- **Asset**: Gold / XAUUSD.
- **Market regime**: Sensitive to real yields, USD, inflation data, central bank
  policy, geopolitical shocks, and ETF/central-bank demand.
- **Data velocity**: Daily/hourly price data for the current demo path; external
  providers should be used before production exposure.
- **Concept drift**: High. A model trained on one macro regime should not be
  treated as stable without monitoring.

### 3.2 Frequency

| Frequency | Trigger | Scope | Resource cost |
| --- | --- | --- | --- |
| Weekly | Scheduled research refresh | Full retrain over a rolling window | Medium |
| Daily | New market/news data available | Lightweight validation and feature refresh | Low |
| Event-driven | Major regime shift or model degradation | Full retrain plus review | High |

Recommended default: weekly full retrain plus event-driven retrain when the
model fails validation.

### 3.3 Monitoring Signals

Track these before promoting a new checkpoint:

- Directional accuracy over the latest holdout window
- Mean absolute error against recent gold prices
- Feature distribution drift versus the training window
- Rate of degraded or heuristic proxy responses from `inference_service.py`
- Freshness and availability of market/news inputs

---

## 4. Promotion and Rollback

1. Train into a temporary output directory, not directly over
   `model_checkpoints/`.
2. Compare the candidate against the current checkpoint on a recent holdout
   window.
3. Promote only if the candidate improves the chosen acceptance metrics and
   preserves the service response contract.
4. Keep the previous checkpoint bundle as a rollback artifact.
5. If live monitoring degrades after promotion, restore the previous bundle and
   mark the failed candidate for review.

Example local retraining command:

```bash
python3 train_stacking.py
```

Example scheduled production shape:

```bash
0 0 * * 0 /usr/bin/python3 /srv/goldensense/train_stacking.py >> /var/log/goldensense/retrain.log 2>&1
```

---

## 5. Validation Checklist

Before publishing new model assets or changing the training path, run:

```bash
python3 -m pytest tests/test_inference_service.py tests/test_agent_analyze.py
python3 scripts/smoke_agent.py
```

For frontend-facing changes to model explanations, also run:

```bash
cd modern_showcase_site
npm run build
npm run test:e2e
```
