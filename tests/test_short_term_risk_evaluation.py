import numpy as np
import pandas as pd
import pytest

from scripts.evaluate_short_term_risk import completed_sessions, summarize_risk


def test_completed_sessions_rejects_missing_close_and_applies_availability_cutoff():
    dates = pd.date_range("2024-01-01", periods=3)
    prices = pd.Series([100, 101, 102], index=dates)
    sessions = pd.DataFrame({
        "Date": dates.strftime("%Y-%m-%d"),
        "close_at": dates.strftime("%Y-%m-%dT18:00:00Z"),
        "available_at": dates.strftime("%Y-%m-%dT20:00:00Z"),
    })
    cutoff = "2024-01-03T19:00:00Z"
    observed = completed_sessions(prices, sessions, through=cutoff)
    assert observed.index.equals(dates[:2])
    with pytest.raises(ValueError, match="calendar"):
        completed_sessions(prices.iloc[1:], sessions, through=cutoff)


def test_risk_summary_uses_same_dates_and_reports_intervals():
    dates = pd.bdate_range("2024-01-01", periods=330)
    rng = np.random.default_rng(8)
    prices = pd.Series(np.exp(5 + rng.normal(0, 0.01, len(dates)).cumsum()), index=dates)
    rows, summary = summarize_risk(prices, horizon=5, min_history=100)
    assert summary["mature_forecasts"] == len(rows) == 225
    for name in ("rolling", "ewma", "har"):
        assert summary["bands"][name]["n"] == 95
        assert rows[f"{name}_p10"].first_valid_index() == rows.index[130]
    lo, hi = summary["har_minus_ewma_block_ci95"]
    assert np.isfinite([lo, hi]).all() and lo <= hi


def test_positive_variance_challengers_use_only_mature_outcomes():
    from vol_models import walk_forward_variance

    dates = pd.bdate_range("2023-01-02", periods=380)
    rng = np.random.default_rng(62)
    prices = pd.Series(np.exp(5 + rng.normal(0, 0.01, len(dates)).cumsum()), index=dates)
    shifted = prices.copy()
    shifted.iloc[300:] *= 1.25
    base = walk_forward_variance(prices, horizon=5, include_challengers=True)
    altered = walk_forward_variance(shifted, horizon=5, include_challengers=True)
    columns = ["log_har_variance", "log_har_vra_variance"]
    pd.testing.assert_frame_equal(base.loc[:dates[299], columns], altered.loc[:dates[299], columns])
    assert (base[columns] > 0).all().all()
    assert not base[columns[0]].equals(base[columns[1]])


def test_student_t_garch_does_not_see_future_prices():
    from garch_risk import walk_forward_garch

    dates = pd.bdate_range("2022-01-03", periods=360)
    rng = np.random.default_rng(62)
    returns = rng.standard_t(df=5, size=len(dates)) * 0.008
    prices = pd.Series(np.exp(5 + returns.cumsum()), index=dates)
    modified = prices.copy()
    modified.iloc[300:] *= 1.2

    observed = walk_forward_garch(prices, horizons=(1, 5), min_history=180)
    revised = walk_forward_garch(modified, horizons=(1, 5), min_history=180)

    pd.testing.assert_frame_equal(observed.loc[:dates[299]], revised.loc[:dates[299]])
    assert (observed.dropna()[5] > observed.dropna()[1]).all()
    assert observed[1].notna().sum() >= 100
