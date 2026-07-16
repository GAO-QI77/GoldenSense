"""Tests for the minimal repo-root .env loader."""
import os

from env_loader import load_env_file


def test_loads_pairs_and_strips_quotes(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment\n"
        "\n"
        "GS_TEST_PLAIN=hello\n"
        'GS_TEST_QUOTED="with spaces"\n'
        "GS_TEST_EQUALS=a=b=c\n"
        "not a pair line\n",
        encoding="utf-8",
    )
    for key in ("GS_TEST_PLAIN", "GS_TEST_QUOTED", "GS_TEST_EQUALS"):
        monkeypatch.delenv(key, raising=False)

    loaded = load_env_file(env_file)

    assert loaded == 3
    assert os.environ["GS_TEST_PLAIN"] == "hello"
    assert os.environ["GS_TEST_QUOTED"] == "with spaces"
    assert os.environ["GS_TEST_EQUALS"] == "a=b=c"
    for key in ("GS_TEST_PLAIN", "GS_TEST_QUOTED", "GS_TEST_EQUALS"):
        monkeypatch.delenv(key, raising=False)


def test_existing_environment_wins(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("GS_TEST_WINNER=from_file\n", encoding="utf-8")
    monkeypatch.setenv("GS_TEST_WINNER", "from_shell")

    loaded = load_env_file(env_file)

    assert loaded == 0
    assert os.environ["GS_TEST_WINNER"] == "from_shell"


def test_missing_file_is_noop(tmp_path):
    assert load_env_file(tmp_path / "does_not_exist.env") == 0
