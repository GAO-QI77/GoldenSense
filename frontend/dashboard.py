from __future__ import annotations

import os
import time
from typing import Dict, List, Optional
from urllib.parse import urlparse

import requests
import streamlit as st
import pandas as pd

from frontend.vix_data import VixDataError, fetch_vix_history, fetch_vix_latest


def _agent_gateway_url() -> str:
    return os.environ.get("AGENT_GATEWAY_URL", "http://localhost:8020/api/v1/agent/trigger")

def _agent_gateway_health_url() -> str:
    u = _agent_gateway_url()
    if u.endswith("/api/v1/agent/trigger"):
        return u[: -len("/api/v1/agent/trigger")] + "/health"
    parsed = urlparse(u)
    return f"{parsed.scheme}://{parsed.netloc}/health"


def _agent_gateway_internal_headers() -> Dict[str, str]:
    return {"X-API-Key": os.environ.get("AGENT_GATEWAY_INTERNAL_API_KEY", "dev-internal-key")}


def _sentiment_color(score: float) -> str:
    if score >= 0.05:
        return "#00d084"
    if score <= -0.05:
        return "#ff4d4f"
    return "#d9d9d9"

def _action_badge(action: str) -> str:
    a = action.upper()
    if a == "BUY":
        return "BUY"
    if a == "SELL":
        return "SELL"
    return "HOLD"

def _action_color(action: str) -> str:
    a = action.upper()
    if a == "BUY":
        return "#00d084"
    if a == "SELL":
        return "#ff4d4f"
    return "#93c5fd"

def _fmt_pct(x: Optional[float]) -> str:
    if x is None:
        return "N/A"
    return f"{x*100:+.2f}%"

def _safe_float(v: object) -> Optional[float]:
    return float(v) if isinstance(v, (int, float)) else None


st.set_page_config(page_title="GoldenSense", layout="wide")

st.markdown(
    """
<style>
  .block-container { padding-top: 1.0rem; }
  html, body, [class*="css"] { background-color: #0b0f14; color: #e6e6e6; }
  .gs-hero { background: radial-gradient(1200px circle at 0% 0%, rgba(0,208,132,0.18), transparent 50%), radial-gradient(1000px circle at 100% 0%, rgba(147,197,253,0.16), transparent 55%), #0b0f14; border: 1px solid #1f2937; border-radius: 18px; padding: 18px; }
  .gs-card { background: linear-gradient(180deg, rgba(17,24,39,0.95), rgba(17,24,39,0.85)); border: 1px solid #1f2937; border-radius: 18px; padding: 16px; }
  .gs-title { font-size: 18px; font-weight: 700; margin-bottom: 6px; letter-spacing: 0.2px; }
  .gs-sub { color: #9ca3af; font-size: 12px; line-height: 1.4; }
  .gs-big { font-size: 42px; font-weight: 800; line-height: 1; }
  .gs-kpi { display:flex; gap: 10px; flex-wrap: wrap; margin-top: 10px; }
  .gs-pill { display:inline-block; padding: 6px 10px; border-radius: 999px; background:#0f172a; border:1px solid #1f2937; font-weight:700; font-size: 12px; }
  .gs-pill-ok { border-color: rgba(0,208,132,0.5); background: rgba(0,208,132,0.08); }
  .gs-pill-warn { border-color: rgba(255,77,79,0.6); background: rgba(255,77,79,0.08); }
  .gs-pill-info { border-color: rgba(147,197,253,0.6); background: rgba(147,197,253,0.08); }
  .gs-banner { border-radius: 16px; padding: 14px 14px; border: 1px solid #1f2937; background: #0f172a; }
  .gs-hr { border: none; border-top: 1px solid #1f2937; margin: 12px 0; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="gs-hero">
  <div style="font-size:22px;font-weight:800;">GoldenSense — Agent 推演面板</div>
  <div class="gs-sub">前端无状态：仅通过 HTTP REST 触发 Agent Gateway；右侧按流水线分步点亮卡片。</div>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### 输入区")
    news_text = st.text_area(
        "突发新闻 (News Text)",
        height=200,
        placeholder="示例：美国劳工部意外大幅上调过去半年的 CPI 数据，通胀二次反弹震惊华尔街。",
    )
    st.markdown("### VIX（实时）")
    auto_refresh = st.checkbox("自动刷新（每 30 秒）", value=False)
    if auto_refresh:
        st.markdown("<meta http-equiv='refresh' content='30'>", unsafe_allow_html=True)

    @st.cache_data(ttl=30, show_spinner=False)
    def _cached_latest_vix() -> dict:
        snap = fetch_vix_latest(timeout_s=6.0)
        return {"value": snap.value, "ts": snap.timestamp.isoformat(), "source": snap.source}

    @st.cache_data(ttl=300, show_spinner=False)
    def _cached_vix_history() -> dict:
        source, pts = fetch_vix_history(range_="6mo", interval="1d", timeout_s=8.0)
        return {
            "source": source,
            "points": [{"ts": p.timestamp.isoformat(), "value": p.value} for p in pts],
        }

    use_live_vix = True
    live_vix_val: Optional[float] = None
    live_vix_ts: Optional[str] = None
    live_vix_source: Optional[str] = None

    try:
        latest = _cached_latest_vix()
        live_vix_val = float(latest["value"])
        live_vix_ts = str(latest["ts"])
        live_vix_source = str(latest["source"])
        st.metric("VIX", f"{live_vix_val:.2f}")
        st.caption(f"来源：{live_vix_source}｜更新时间：{live_vix_ts}")
    except Exception:
        use_live_vix = False
        st.warning("实时 VIX 获取失败，将使用手动滑块作为回退。")

    manual_vix = st.slider("手动 VIX（回退/覆盖）", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
    vix_for_trigger = live_vix_val if (use_live_vix and live_vix_val is not None) else float(manual_vix)

    if vix_for_trigger >= 30.0:
        st.markdown(f'<div class="gs-pill gs-pill-warn">熔断红线：30.0（已触发，当前 {vix_for_trigger:.1f}）</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="gs-pill gs-pill-info">熔断红线：30.0（当前 {vix_for_trigger:.1f}）</div>', unsafe_allow_html=True)

    show_vix_history = st.checkbox("显示 VIX 历史趋势", value=True)
    trigger = st.button("Trigger Agent (执行推演)", use_container_width=True, type="primary")

top_left, top_right = st.columns([1, 2], gap="large")

with top_left:
    health_ok = False
    try:
        r = requests.get(_agent_gateway_health_url(), timeout=2.5)
        health_ok = r.status_code == 200
    except Exception:
        health_ok = False

    status_pill = "ONLINE" if health_ok else "OFFLINE"
    status_class = "gs-pill-ok" if health_ok else "gs-pill-warn"
    st.markdown(
        f"""
<div class="gs-card">
  <div class="gs-title">连接状态</div>
  <div class="gs-sub">Agent Gateway</div>
  <div class="gs-kpi">
    <span class="gs-pill {status_class}">{status_pill}</span>
    <span class="gs-pill gs-pill-info">{_agent_gateway_url()}</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    if show_vix_history:
        try:
            hist = _cached_vix_history()
            pts = hist.get("points", [])
            if isinstance(pts, list) and pts:
                df = pd.DataFrame(pts)
                df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
                df = df.dropna().sort_values("ts")
                df = df.set_index("ts")
                st.line_chart(df["value"], height=160)
        except Exception:
            st.caption("VIX 历史趋势暂不可用。")


def _render_pipeline(data: Dict[str, object]) -> None:
    decision = data.get("decision", {})
    risk_result = data.get("risk_result", {})

    finbert_score = float(data.get("finbert_sentiment_score", 0.0))
    rag_titles = data.get("rag_top_3_event_titles", [])
    rag_events = data.get("rag_top_3_events", [])
    xgb_prob_raw = data.get("xgboost_probability")
    quant_prob_raw = data.get("quant_probability")
    xgb_prob = _safe_float(xgb_prob_raw)
    quant_prob = _safe_float(quant_prob_raw)

    color = _sentiment_color(finbert_score)

    with st.status("卡片 1 / 感知层：情绪得分", expanded=True):
        time.sleep(0.15)
        st.markdown(
            f"""
<div class="gs-card">
  <div class="gs-title">感知层 (Perception)</div>
  <div class="gs-sub">FinBERT Sentiment Score (P(Pos)-P(Neg))</div>
  <div class="gs-big" style="color:{color};">{finbert_score:+.3f}</div>
  <div class="gs-sub">解释：数值越接近 +1 越偏多；越接近 -1 越偏空。</div>
</div>
""",
            unsafe_allow_html=True,
        )

    with st.status("卡片 2 / 记忆层：历史相似事件", expanded=False):
        time.sleep(0.15)
        st.markdown('<div class="gs-card"><div class="gs-title">记忆层 (Memory / RAG)</div><div class="gs-sub">Top-K 相似事件剧本与真实后验收益</div></div>', unsafe_allow_html=True)
        with st.expander("Top-3 相似事件（展开查看）", expanded=True):
            if isinstance(rag_events, list) and rag_events:
                for i, e in enumerate(rag_events[:3], start=1):
                    if isinstance(e, dict):
                        title = str(e.get("headline", ""))
                        sim = _safe_float(e.get("similarity"))
                        t1 = _safe_float(e.get("gold_t1_return"))
                        t7 = _safe_float(e.get("gold_t7_return"))
                        sim_s = f"{sim:.3f}" if sim is not None else "N/A"
                        st.markdown(f"- **#{i}** {title}  \n  相似度：{sim_s}｜T+1：{_fmt_pct(t1)}｜T+7：{_fmt_pct(t7)}")
            elif isinstance(rag_titles, list) and rag_titles:
                for i, t in enumerate(rag_titles[:3], start=1):
                    st.markdown(f"- **#{i}** {t}")
            else:
                st.markdown("- 未获取到相似事件（上游服务可能不可用）")

    with st.status("卡片 3 / 量化层：XGBoost 胜率", expanded=False):
        time.sleep(0.15)
        st.markdown(
            """
<div class="gs-card">
  <div class="gs-title">量化层 (Quant)</div>
  <div class="gs-sub">XGBoost Probability & Ensemble Probability</div>
</div>
""",
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns(2)
        with c1:
            if xgb_prob is None:
                st.metric("XGBoost 概率", "N/A")
                st.progress(0.0)
            else:
                st.metric("XGBoost 概率", f"{xgb_prob:.3f}")
                st.progress(min(max(xgb_prob, 0.0), 1.0))
        with c2:
            if quant_prob is None:
                st.metric("Ensemble 概率", "N/A")
                st.progress(0.0)
            else:
                st.metric("Ensemble 概率", f"{quant_prob:.3f}")
                st.progress(min(max(quant_prob, 0.0), 1.0))

    with st.status("卡片 4 / 认知层：Reasoning Summary", expanded=False):
        time.sleep(0.15)
        reasoning = ""
        if isinstance(decision, dict):
            reasoning = str(decision.get("reasoning_summary", ""))
        st.markdown(
            f"""
<div class="gs-card">
  <div class="gs-title">认知层 (Cognitive)</div>
  <blockquote>{reasoning}</blockquote>
</div>
""",
            unsafe_allow_html=True,
        )

    with st.status("卡片 5 / 风控层：终极裁决", expanded=True):
        time.sleep(0.15)
        action = ""
        conf = 0.0
        horizon = ""
        if isinstance(decision, dict):
            action = str(decision.get("action", "HOLD"))
            conf = float(decision.get("confidence", 0.0))
            horizon = str(decision.get("horizon", "T+1"))

        risk_decision = ""
        executed_pos = 0.0
        current_vix = None
        vix_threshold = None
        notes = ""
        if isinstance(risk_result, dict):
            risk_decision = str(risk_result.get("decision", ""))
            executed_pos = float(risk_result.get("executed_position", 0.0))
            current_vix = risk_result.get("current_vix")
            vix_threshold = risk_result.get("vix_threshold")
            notes = str(risk_result.get("notes", ""))

        st.markdown(
            f"""
<div class="gs-card">
  <div class="gs-title">风控层 (Override Layer)</div>
  <div class="gs-sub">Action={action} Confidence={conf:.2f} Horizon={horizon}</div>
</div>
""",
            unsafe_allow_html=True,
        )

        banner_color = _action_color(action)
        st.markdown(
            f"""
<div class="gs-banner">
  <div style="font-size:34px;font-weight:900;color:{banner_color};">{_action_badge(action)}</div>
  <div class="gs-sub">Risk Decision: {risk_decision} ｜ Estimated Position: {executed_pos} lots</div>
</div>
""",
            unsafe_allow_html=True,
        )

        if risk_decision in {"REJECTED", "EXEC_FAILED"}:
            if current_vix is not None and vix_threshold is not None:
                st.error(f"REJECTED — VIX 触发熔断 ({current_vix} > {vix_threshold})")
            else:
                st.error(f"{risk_decision} — {notes}")
        elif risk_decision == "EXECUTED":
            st.success(f"EXECUTED — 预估头寸: {executed_pos} 手")
        else:
            st.info(f"{risk_decision} — {notes}")


with top_right:
    st.markdown("### 流水线 (5 Cards)")
    if trigger:
        if not news_text.strip():
            st.error("请输入突发新闻文本。")
        else:
            payload = {"news_text": news_text.strip(), "manual_vix": manual_vix}
            try:
                with st.spinner("系统连接中：正在触发 Agent Gateway..."):
                    resp = requests.post(
                        _agent_gateway_url(),
                        json={"news_text": news_text.strip(), "manual_vix": vix_for_trigger},
                        headers=_agent_gateway_internal_headers(),
                        timeout=60,
                    )
                impact_header = resp.headers.get("X-Impact-Breakdown")
                if impact_header:
                    with st.expander("Impact Breakdown（来自响应头 X-Impact-Breakdown）", expanded=True):
                        st.code(impact_header, language="json")

                if resp.status_code != 200:
                    st.error("系统连接中断：Agent Gateway 返回异常。")
                    st.code(resp.text)
                else:
                    data = resp.json()
                    _render_pipeline(data)
                    with st.expander("原始 JSON（用于审计/排错）", expanded=False):
                        st.json(data)
            except Exception:
                st.error("系统连接中断：无法连接到 Agent Gateway。")
    else:
        st.markdown(
            '<div class="gs-card"><div class="gs-sub">在左侧输入新闻与 VIX，点击 Trigger Agent 开始推演。</div></div>',
            unsafe_allow_html=True,
        )
