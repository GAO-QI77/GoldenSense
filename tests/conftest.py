"""Test-session guards.

A developer's repo-root .env may hold real LLM keys (loaded by env_loader on
gateway import). Tests must never place live LLM calls -- they would be slow,
flaky, and cost money -- so the keys are stripped before any test imports the
gateway. Narrator behaviour in tests is exercised via injected stubs only.
"""
import os

os.environ.pop("DEEPSEEK_API_KEY", None)
os.environ.pop("OPENAI_API_KEY", None)
# Ensure the loader cannot re-populate them mid-session: an empty value in
# the environment takes precedence over the .env file by design.
os.environ["DEEPSEEK_API_KEY"] = ""
os.environ["OPENAI_API_KEY"] = ""

# No background daemon threads in tests: every create_app() would otherwise
# spawn a quant-context warm-up thread that outlives its test and can deadlock
# a later test's event-loop shutdown (asyncio waits on its default executor;
# executor callables can contend with the warm-up thread). Context is still
# computed lazily on-demand inside request paths, so nothing is lost.
os.environ["AGENT_WARM_QUANT_CONTEXT"] = "0"
os.environ["SIGNAL_AUTOPUBLISH_ENABLED"] = "0"

# Tests control checkpoint locations explicitly; a developer .env pointing at
# the real model_checkpoints/ directory once made a "missing checkpoint" test
# load a real torch model inside an executor thread and deadlock the event
# loop shutdown. Strip the overrides so the suite is .env-independent.
os.environ.pop("INFERENCE_MODEL_CHECKPOINTS_DIR_T1", None)
os.environ.pop("INFERENCE_MODEL_CHECKPOINTS_DIR_T7", None)
