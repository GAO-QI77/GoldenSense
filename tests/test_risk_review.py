from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

from risk_review import CANDIDATES, WINDOWS, read_risk_review


def test_read_review_respects_generation_cutoff_and_code_version(tmp_path):
    path = tmp_path / "report.json"
    digest = sha256((Path(__file__).parents[1] / "vol_models.py").read_bytes()).hexdigest()
    report = {
        "protocol": "daily_close_variance_baselines_v3",
        "instrument": "GC=F",
        "generated_at": "2026-11-10T06:00:00+00:00",
        "price_end": "2026-11-09",
        "model_code_sha256": digest,
        "windows": {
            w: {
                "release_review": {m: {"status": "review_required", "blockers": []} for m in CANDIDATES},
                "bands": {m: {"n": 140, "coverage": 0.8} for m in CANDIDATES},
            }
            for w in WINDOWS
        },
    }
    path.write_text(json.dumps(report), encoding="utf-8")
    now = datetime(2026, 11, 10, 7, tzinfo=timezone.utc)

    assert read_risk_review(path, decision_at=now.replace(hour=5), data_asof="2026-11-09") is None
    assert read_risk_review(path, decision_at=now, data_asof="2026-11-06") is None
    result = read_risk_review(path, decision_at=now, data_asof="2026-11-09")
    assert all(candidate["status"] == "hold" for w in result["windows"].values() for candidate in w.values())
    from risk_review import GARCH_CANDIDATE, garch_model_hash

    report["protocol"] = "daily_close_variance_baselines_v4"
    report["garch"] = {"code_sha256": garch_model_hash()}
    for window in report["windows"].values():
        window["release_review"][GARCH_CANDIDATE] = {"status": "hold", "blockers": ["incomplete_forecasts"]}
        window["bands"][GARCH_CANDIDATE] = {"n": 0}
    path.write_text(json.dumps(report), encoding="utf-8")
    result = read_risk_review(path, decision_at=now, data_asof="2026-11-09")
    assert result["windows"]["1"][GARCH_CANDIDATE]["blockers"] == ["incomplete_forecasts"]
    report["garch"]["code_sha256"] = "wrong"
    path.write_text(json.dumps(report), encoding="utf-8")
    assert read_risk_review(path, decision_at=now, data_asof="2026-11-09") is None
    report["garch"]["code_sha256"] = garch_model_hash()
    report["model_code_sha256"] = "different"
    path.write_text(json.dumps(report), encoding="utf-8")
    assert read_risk_review(path, decision_at=now, data_asof="2026-11-09") is None
