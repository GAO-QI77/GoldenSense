"""Validate the local challenger report before surfacing it in research views."""
from datetime import date, datetime
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path

CANDIDATES = ("log_har", "log_har_vra")
WINDOWS = ("1", "5", "21")
BLOCKERS = {
    "session_timestamps_missing", "evaluation_start_missing", "insufficient_mature_bands",
    "coverage_or_tail_failed", "incremental_error_not_demonstrated",
}


def read_risk_review(path: Path, *, decision_at: datetime, data_asof: str) -> dict | None:
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        model_digest = sha256(Path(__file__).with_name("vol_models.py").read_bytes()).hexdigest()
        generated_at = datetime.fromisoformat(raw["generated_at"])
        if (
            raw["protocol"] != "daily_close_variance_baselines_v3"
            or raw["instrument"] != "GC=F"
            or raw["model_code_sha256"] != model_digest
            or generated_at > decision_at
            or date.fromisoformat(raw["price_end"]) > date.fromisoformat(data_asof)
        ):
            return None
        windows = {}
        for horizon in WINDOWS:
            window = raw["windows"][horizon]
            windows[horizon] = {}
            for model in CANDIDATES:
                report = window["release_review"][model]
                bands = window["bands"][model]
                flags = report["blockers"]
                coverage = bands.get("coverage")
                if (
                    report["status"] not in ("hold", "review_required")
                    or not isinstance(flags, list) or not set(flags) <= BLOCKERS
                    or type(bands["n"]) is not int or bands["n"] < 0
                    or (coverage is not None and not (isfinite(coverage) and 0 <= coverage <= 1))
                ):
                    return None
                windows[horizon][model] = {
                    "status": "hold" if flags or not raw.get("calendar_sha256") else report["status"],
                    "blockers": flags, "n": bands["n"], "coverage": coverage,
                }
        return {"instrument": "GC=F", "price_end": raw["price_end"], "windows": windows}
    except (OSError, ValueError, TypeError, KeyError):
        return None
