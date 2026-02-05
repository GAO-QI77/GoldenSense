import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import time
from datetime import datetime, timedelta

# --- 1. 配置与多语言支持 ---
st.set_page_config(page_title="GoldenSense | Live AI Market Monitor", layout="wide", initial_sidebar_state="collapsed")

# 自动刷新机制 (每 60 秒刷新一次)
from streamlit.runtime.scriptrunner import add_script_run_ctx
if 'last_update' not in st.session_state:
    st.session_state.last_update = time.time()

# 自定义 CSS (保持不变)
st.markdown("""
<style>
    .main { background-color: #0e1117; color: #fafafa; }
    .stMetric { background-color: #262730; padding: 15px; border-radius: 10px; border: 1px solid #41424b; }
    .prediction-card { padding: 20px; border-radius: 15px; color: white; margin-bottom: 20px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
    .card-up { background: linear-gradient(135deg, #00c853, #007e33); }
    .card-down { background: linear-gradient(135deg, #ff3547, #cc0000); }
    .section-header { border-left: 5px solid #ffc107; padding-left: 15px; margin: 30px 0 20px 0; font-weight: bold; color: #ffc107; }
    h1, h2, h3, h4 { color: #fafafa; }
</style>
""", unsafe_allow_html=True)

LANG = {
    "CN": {
        "title": "GoldenSense - 黄金市场实时智能监控",
        "prediction_area": "AI 实时趋势预测 (Live Inference)",
        "market_analysis": "实时市场情报流",
        "news_info": "全球快讯",
        "policy_info": "央行动态",
        "geopolitics_info": "地缘雷达",
        "analysis_report": "AI 深度决策简报",
        "aux_content": "多维市场数据",
        "market_info": "实时行情",
        "factors": "宏观因子",
        "tech_analysis": "技术信号",
        "history": "走势回溯",
        "tomorrow": "T+1 (24h)",
        "week": "T+7 (Weekly)",
        "month": "T+30 (Monthly)",
        "up": "看涨 Bullish",
        "down": "看跌 Bearish",
        "prob": "置信度",
        "trend": "预期波幅",
        "realtime_price": "现货黄金 (XAU/USD)",
        "disclaimer": "⚠️ 本系统数据仅供参考，不构成投资建议。市场有风险，投资需谨慎。",
        "refresh": "上次更新时间"
    },
    "EN": {
        "title": "GoldenSense - Live AI Market Monitor",
        "prediction_area": "AI Trend Prediction (Live)",
        "market_analysis": "Real-time Market Intelligence",
        "news_info": "Global Breaking",
        "policy_info": "Central Banks",
        "geopolitics_info": "Geopolitics",
        "analysis_report": "AI Decision Brief",
        "aux_content": "Multi-dimensional Data",
        "market_info": "Live Quotes",
        "factors": "Macro Factors",
        "tech_analysis": "Technical Signals",
        "history": "Price History",
        "tomorrow": "T+1 (24h)",
        "week": "T+7 (Weekly)",
        "month": "T+30 (Monthly)",
        "up": "Bullish",
        "down": "Bearish",
        "prob": "Confidence",
        "trend": "Exp. Volatility",
        "realtime_price": "Spot Gold (XAU/USD)",
        "disclaimer": "⚠️ Data for reference only. Not financial advice.",
        "refresh": "Last Updated"
    }
}

# --- 2. 实时数据获取函数 ---
@st.cache_data(ttl=60) # 缓存 60 秒，避免频繁请求被封
def get_live_market_data():
    tickers = {
        'Gold': 'GC=F', 'Silver': 'SI=F', 'USD_Index': 'DX=F', 
        'S&P500': '^GSPC', 'Crude_Oil': 'CL=F', 'VIX': '^VIX', '10Y_Bond': '^TNX'
    }
    data = {}
    try:
        # 批量下载，获取最近 5 天数据以计算涨跌幅
        raw = yf.download(list(tickers.values()), period="5d", interval="1d", progress=False)['Close']
        
        # 处理多层列索引问题
        if isinstance(raw.columns, pd.MultiIndex):
            # 展平列名，只保留 ticker 部分
            raw.columns = raw.columns.get_level_values(0)
            
        for name, ticker in tickers.items():
            if ticker in raw.columns:
                series = raw[ticker].dropna()
                if len(series) >= 2:
                    data[name] = {
                        'price': series.iloc[-1],
                        'change': (series.iloc[-1] - series.iloc[-2]) / series.iloc[-2]
                    }
                    data[f'{name}_History'] = series # 保存历史序列用于绘图
    except Exception as e:
        st.error(f"Data fetch error: {e}")
    return data

# --- 3. 轻量级在线推断逻辑 (模拟) ---
def run_live_inference(market_data):
    # 这里模拟 AI 模型的实时输出
    # 实际部署时，这里应加载 pytorch 模型进行 forward pass
    # 为了演示效果，我们使用技术指标作为推断依据
    
    gold_change = market_data.get('Gold', {}).get('change', 0)
    usd_change = market_data.get('USD_Index', {}).get('change', 0)
    vix_price = market_data.get('VIX', {}).get('price', 15)
    
    # 简易逻辑：美元跌 & 恐慌升 -> 黄金涨
    score = -usd_change * 5 + (vix_price - 15) * 0.02 + gold_change * 2
    prob = 0.5 + np.tanh(score) * 0.4
    
    return {
        '1d': {'dir': 1 if score > 0 else -1, 'prob': prob, 'vol': abs(score) * 0.01 + 0.005},
        '7d': {'dir': 1 if score > -0.1 else -1, 'prob': prob * 0.9, 'vol': abs(score) * 0.03 + 0.015},
        '30d': {'dir': 1, 'prob': 0.65, 'vol': 0.05} # 长期看涨假设
    }

# --- 主程序 ---
lang_choice = st.sidebar.radio("Language / 语言", ["CN", "EN"])
t = LANG[lang_choice]

# 标题栏
c_title, c_time = st.columns([3, 1])
with c_title:
    st.title(f"🚀 {t['title']}")
with c_time:
    st.caption(f"{t['refresh']}: {datetime.now().strftime('%H:%M:%S')}")
    if st.button("🔄 Refresh"):
        st.cache_data.clear()
        st.rerun()

st.info(t['disclaimer'])

# 获取实时数据
live_data = get_live_market_data()

if not live_data:
    st.warning("Market data is initializing... Please refresh.")
    st.stop()

# 运行实时推断
predictions = run_live_inference(live_data)

# --- 5. 核心价格预测展示 ---
st.markdown(f"<div class='section-header'><h3>🔮 {t['prediction_area']}</h3></div>", unsafe_allow_html=True)

cols = st.columns(3)
horizons = ['1d', '7d', '30d']
labels = [t['tomorrow'], t['week'], t['month']]

for i, h in enumerate(horizons):
    with cols[i]:
        pred = predictions[h]
        is_up = pred['dir'] > 0
        card_class = "card-up" if is_up else "card-down"
        dir_text = t['up'] if is_up else t['down']
        
        st.markdown(f"""
        <div class="prediction-card {card_class}">
            <h4>{labels[i]}</h4>
            <h2 style="margin: 10px 0;">{dir_text}</h2>
            <p>{t['prob']}: {pred['prob']*100:.1f}%</p>
            <p>{t['trend']}: {pred['vol']*100:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)

# --- 6. 辅助决策内容 (已调换位置至此) ---
st.markdown(f"<div class='section-header'><h3>🧩 {t['aux_content']}</h3></div>", unsafe_allow_html=True)

tabs = st.tabs([t['market_info'], t['factors'], t['tech_analysis'], t['history']])

with tabs[0]: # 实时行情
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(t['realtime_price'], f"${live_data['Gold']['price']:,.2f}", f"{live_data['Gold']['change']*100:+.2f}%")
    c2.metric("Silver", f"${live_data['Silver']['price']:.2f}", f"{live_data['Silver']['change']*100:+.2f}%")
    c3.metric("USD Index", f"{live_data['USD_Index']['price']:.2f}", f"{live_data['USD_Index']['change']*100:+.2f}%")
    c4.metric("Crude Oil", f"${live_data['Crude_Oil']['price']:.2f}", f"{live_data['Crude_Oil']['change']*100:+.2f}%")

with tabs[1]: # 关键因子
    c1, c2 = st.columns(2)
    with c1:
        st.write("#### 10Y Treasury Yield")
        bond_hist = live_data.get('10Y_Bond_History')
        if bond_hist is not None:
            st.line_chart(bond_hist)
    with c2:
        st.write("#### VIX (Volatility Index)")
        vix_hist = live_data.get('VIX_History')
        if vix_hist is not None:
            st.line_chart(vix_hist)

with tabs[2]: # 技术分析
    pivot = live_data['Gold']['price']
    st.table(pd.DataFrame({
        "Level": ["R2", "R1", "Pivot", "S1", "S2"],
        "Price": [f"${pivot*1.02:.2f}", f"${pivot*1.01:.2f}", f"${pivot:.2f}", f"${pivot*0.99:.2f}", f"${pivot*0.98:.2f}"]
    }, index=["强阻力", "弱阻力", "中枢", "弱支撑", "强支撑"]))

with tabs[3]: # 历史价格
    gold_hist = live_data.get('Gold_History')
    if gold_hist is not None:
        fig = px.line(gold_hist, title="Spot Gold Price Trend (Recent)")
        fig.update_traces(line_color='#ffc107')
        st.plotly_chart(fig, use_container_width=True)

# --- 7. 综合分析区域 (已调换位置至此) ---
st.markdown(f"<div class='section-header'><h3>📊 {t['market_analysis']}</h3></div>", unsafe_allow_html=True)

analysis_cols = st.columns([1, 1, 1])
with analysis_cols[0]:
    st.info(f"📰 **{t['news_info']}**")
    st.markdown("""
    - 🔴 **Breaking**: US Core PCE data exceeds expectations
    - Gold prices test $2350 support level
    - Central banks continue record buying spree
    """)
with analysis_cols[1]:
    st.warning(f"🏛️ **{t['policy_info']}**")
    st.markdown("""
    - **Fed**: Powell hints at "higher for longer" rates
    - **ECB**: Rate cut likely in June
    - **BoJ**: Maintaining ultra-loose policy
    """)
with analysis_cols[2]:
    st.error(f"🌍 **{t['geopolitics_info']}**")
    st.markdown("""
    - Middle East tensions remain elevated
    - Trade supply chain disruptions
    - Safe-haven demand spiking
    """)

# 深度简报 (颜色已调整为浅色主题)
st.markdown(f"#### 🧠 {t['analysis_report']}")
with st.container():
    st.markdown(f"""
    <div style="background-color: #f8f9fa; padding: 25px; border-radius: 12px; border-left: 6px solid #d97706; box-shadow: 0 2px 5px rgba(0,0,0,0.05); color: #1f2937;">
        <p style="margin-bottom: 15px;"><strong style="color: #b45309; font-size: 1.1em;">AI 实时研判：</strong> 当前市场正处于关键的宏观数据发布窗口期。实时数据显示美元指数（DXY）微幅震荡，而黄金（XAUUSD）展现出极强的抗跌性。模型识别到大量避险资金流入，抵消了高利率环境的压力。</p>
        <p style="margin-bottom: 15px;"><strong style="color: #b45309; font-size: 1.1em;">关键信号：</strong> 10年期美债收益率与金价的负相关性近期有所减弱，这表明地缘溢价正在成为主导因子。技术面上，金价站稳 20日均线，多头动能正在积蓄。</p>
        <p style="margin-top: 15px; padding-top: 10px; border-top: 1px solid #e5e7eb;"><strong style="color: #dc2626; font-size: 1.1em;">操作建议：</strong> 建议重点关注今日收盘价。若能突破关键阻力位，T+1 模型预测将迎来一波快速拉升。短线交易者可逢低做多，长线投资者继续持有。</p>
    </div>
    """, unsafe_allow_html=True)
