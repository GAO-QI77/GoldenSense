"""Deterministic toolbox + narrator for the eval harness.

Reuses the proven scenario toolbox from the test-suite (a dev/CI dependency,
never shipped to the production runtime) so the eval gate drives the real
analyze pipeline without network, downstream services, or an LLM key.
"""
from __future__ import annotations

import os
import sys

_TESTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests")
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from test_agent_analyze import _DraftNarrator, _ScenarioToolbox  # noqa: E402


class StubToolbox(_ScenarioToolbox):
    """Alias with a stable public name for the eval harness."""


class StubNarrator(_DraftNarrator):
    """Deterministic narrator: returns the rule-based draft unchanged."""
