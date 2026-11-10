
import pytest

from scripts import refresh_macro_vintages as refresh


def test_paginated_macro_import_delays_date_only_versions(monkeypatch, tmp_path):
    monkeypatch.setattr(refresh, "FRED_SERIES", {"Real_10Y": "DFII10"})
    calls = []

    def fetch(params):
        offset = params["offset"]
        calls.append(offset)
        assert params["output_type"] == 1
        return {
            "count": 2,
            "observations": [{
                "date": "2024-03-08",
                "realtime_start": ["2024-03-08", "2024-03-11"][offset],
                "value": [1.5, 1.7][offset],
            }],
        }

    dest = tmp_path / "archive.csv"
    assert refresh.refresh_archive(dest, fetch) == {"DFII10": 2}
    assert calls == [0, 1]
    rows = dest.read_text().splitlines()
    assert rows[1] == "DFII10,2024-03-08,2024-03-09T05:00:00+00:00,1.5"
    assert rows[2] == "DFII10,2024-03-08,2024-03-12T04:00:00+00:00,1.7"


def test_partial_download_preserves_last_good_archive(monkeypatch, tmp_path):
    monkeypatch.setattr(refresh, "FRED_SERIES", {"Real_10Y": "DFII10"})
    dest = tmp_path / "archive.csv"
    dest.write_text("existing\n")

    def fetch(params):
        return {"count": 1, "observations": []}

    with pytest.raises(ValueError, match="incomplete"):
        refresh.refresh_archive(dest, fetch)
    assert dest.read_text() == "existing\n"


def test_imports_verified_public_alfred_exports(monkeypatch, tmp_path):
    from hashlib import sha256
    import json
    from zipfile import ZipFile

    monkeypatch.setattr(refresh, "FRED_SERIES", {"Real_10Y": "DFII10"})
    source = tmp_path / "DFII10.zip"
    with ZipFile(source, "w") as archive:
        archive.writestr(
            "obs._by_real-time_period.csv",
            "period_start_date,DFII10,realtime_start_date,realtime_end_date\n"
            "2024-03-08,1.5,2024-03-08,2024-03-10\n"
            "2024-03-08,1.7,2024-03-11,\n",
        )
    manifest = {
        "DFII10": {"batches": [{
            "file": source.name, "rows": 2, "sha256": sha256(source.read_bytes()).hexdigest(),
        }]},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    dest = tmp_path / "macro.csv"
    assert refresh.import_exports(dest, tmp_path) == {"DFII10": 2}
    assert dest.read_text().splitlines()[-1] == "DFII10,2024-03-08,2024-03-12T04:00:00+00:00,1.7"
    source.write_bytes(b"damaged")
    with pytest.raises(ValueError, match="checksum"):
        refresh.import_exports(dest, tmp_path)
    assert "1.7" in dest.read_text()
