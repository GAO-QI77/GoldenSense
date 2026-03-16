import uvicorn
from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security.api_key import APIKeyHeader
import asyncio
import json
import random
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add parent directory to sys.path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stacking_model import DynamicEnsemble
from data_loader import MarketDataLoader, NewsDataLoader
from feature_engineer import FeatureEngineer

app = FastAPI()

# 0. Security & Performance
API_KEY_NAME = "X-API-Key"
API_KEY = os.getenv("API_KEY", "demo-key-123") # Default for demo if not set
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key: str = Depends(api_key_header)):
    # Allow public access for now as per requirement, but structure is here
    # If we wanted to enforce:
    # if api_key != API_KEY:
    #     raise HTTPException(status_code=403, detail="Could not validate credentials")
    return api_key

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Should be restricted in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# 2. Static Files
app.mount("/static", StaticFiles(directory="live_platform/static"), name="static")

# Global State
market_state = {
    "price": 2350.0,
    "timestamp": datetime.now().isoformat(),
    "history": [],
    "prediction": {
        "1d": {"dir": 0, "prob": 0.5, "vol": 0.0},
        "7d": {"dir": 0, "prob": 0.5, "vol": 0.0},
        "30d": {"dir": 0, "prob": 0.5, "vol": 0.0}
    },
    "news": []
}

# Initialize Components
market_loader = MarketDataLoader()
news_loader = NewsDataLoader()
feature_engineer = FeatureEngineer()
# Initialize model with correct dimensions used in training
# Tabular: 15 selected features
# Sequence: 4 price features (Gold, Silver, Crude, USD)
model = DynamicEnsemble(tabular_input_dim=15, seq_input_dim=4) 

# Try to load model
if os.path.exists("model_checkpoints"):
    model.load_model("model_checkpoints")
    print("Loaded pre-trained model.")
else:
    print("Warning: No pre-trained model found. Predictions will be initialized.")

async def update_system_state():
    """Background task to fetch data and run inference"""
    while True:
        try:
            print("Updating system state...")
            
            # 1. Fetch Market Data (Retry logic is in data_loader)
            raw_data = market_loader.fetch_data(period='2y', interval='1d')
            
            if not raw_data.empty:
                # Update Price Cache
                last_price = float(raw_data['Gold'].iloc[-1])
                market_state["price"] = last_price
                market_state["timestamp"] = datetime.now().isoformat()
                market_state["history"] = raw_data['Gold'].tail(50).tolist()
                
                # 2. Fetch & Analyze News
                news_items = news_loader.fetch_news()
                scored_news = news_loader.analyze_causality(news_items)
                
                # Update News Cache (Top 5)
                if scored_news:
                    scored_news.sort(key=lambda x: abs(x['total']), reverse=True)
                    market_state["news"] = [{
                        "time": item['published'],
                        "title": item['title'],
                        "category": "High Impact" if abs(item['total']) > 1.5 else "General",
                        "score": item['total']
                    } for item in scored_news[:5]]
                
                daily_signals = news_loader.get_daily_signals(scored_news)
                
                # 3. Feature Engineering
                # We need a sequence for GRU/Transformer, so we need the last N days
                # FeatureEngineer.prepare_inference_data returns the last row of features
                # But DynamicEnsemble needs (X_tab, X_seq)
                
                # We need to run the full pipeline on recent data to generate the sequence
                # Let's run pipeline on the last 90 days to ensure we have enough for seq_length=60
                recent_data = raw_data.tail(120) 
                
                # We reuse run_pipeline logic but we need to extract X_tab and X_seq for the *last* step
                # Hack: FeatureEngineer.run_pipeline returns X (DataFrame).
                # We can use that.
                
                X_features = feature_engineer.prepare_inference_data(recent_data, daily_signals, n_rows=60)
                
                if not X_features.empty and len(X_features) >= 60:
                    # Prepare inputs for model
                    
                    # X_tab: Last row, selected features only
                    # Load selected features if available
                    selected_features = feature_engineer.load_selected_features()
                    if not selected_features:
                        # Fallback if no file, use all or heuristic
                        print("Warning: selected_features.json not found. Using all features.")
                        X_tab = X_features.iloc[[-1]].values
                        # Adjust model input dim if needed (but model is already loaded with fixed dim)
                    else:
                        # Ensure all selected features exist
                        available_features = [f for f in selected_features if f in X_features.columns]
                        if len(available_features) < len(selected_features):
                            print(f"Warning: Missing features. Expected {len(selected_features)}, found {len(available_features)}")
                        X_tab = X_features[available_features].iloc[[-1]].values

                    # X_seq: Last 60 rows, specific columns (ZScore versions)
                    # Train script used: seq_cols = ['Gold_ZScore', 'Silver_ZScore', 'Crude_Oil_ZScore', 'USD_Index_ZScore']
                    seq_cols = ['Gold_ZScore', 'Silver_ZScore', 'Crude_Oil_ZScore', 'USD_Index_ZScore']
                    # Check if columns exist
                    valid_seq_cols = [c for c in seq_cols if c in X_features.columns]
                    if len(valid_seq_cols) == 4:
                        X_seq = X_features[valid_seq_cols].values.reshape(1, 60, -1)
                    else:
                        print(f"Warning: Missing sequence columns. Found: {valid_seq_cols}")
                        # Fallback: try original names if ZScore missing (unlikely given prepare_inference_data calls adaptive_normalize)
                        X_seq = X_features.iloc[-60:, :4].values.reshape(1, 60, -1)
                    
                    # 4. Run Inference
                    # We predict for 1d, 7d, 30d (logic needs to be adapted as model predicts one target at a time normally)
                    # For this demo, we will use the single model instance to predict '1d' 
                    # (assuming the loaded model was trained for 1d, or we use the raw output as a trend indicator)
                    
                    pred_val = model.predict(X_tab, X_seq)
                    
                    # Interpret prediction (assuming return is float, e.g., log return or z-score)
                    # Simple heuristic for demo if model output is raw return
                    pred_return = pred_val[0] if isinstance(pred_val, (np.ndarray, list)) else pred_val
                    
                    # Simulate multi-horizon confidence based on the single prediction intensity
                    prob = 0.5 + np.tanh(pred_return * 10) * 0.4 
                    
                    market_state["prediction"] = {
                        "1d": {"dir": 1 if pred_return > 0 else -1, "prob": abs(prob), "vol": abs(pred_return)*5},
                        "7d": {"dir": 1 if pred_return > 0 else -1, "prob": abs(prob)*0.9, "vol": abs(pred_return)*10},
                        "30d": {"dir": 1 if pred_return > 0 else -1, "prob": abs(prob)*0.8, "vol": abs(pred_return)*20}
                    }
                    print(f"Inference run: Pred={pred_return:.5f}")

        except Exception as e:
            print(f"Error in update loop: {e}")
            import traceback
            traceback.print_exc()
            
        await asyncio.sleep(60) # Update every minute

# 3. Routes
@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("live_platform/static/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/api/history")
async def get_history(api_key: str = Depends(get_api_key)):
    return {"history": market_state["history"], "current": market_state["price"]}

@app.get("/api/stream")
async def message_stream(request: Request):
    """Server-Sent Events (SSE) for real-time price updates"""
    async def event_generator():
        while True:
            if await request.is_disconnected():
                break
            
            # Use REAL data from market_state
            current_price = market_state["price"]
            # Calculate a micro-fluctuation based on volatility for "liveness" feel between 1-min updates
            # (Optional: can just send current_price)
            # noise = random.gauss(0, 0.1) 
            
            data = json.dumps({
                "price": round(current_price, 2),
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "change_percent": (current_price - market_state["history"][-2]) / market_state["history"][-2] if len(market_state["history"]) > 1 else 0
            })
            yield f"data: {data}\n\n"
            await asyncio.sleep(3) # Optimized frequency (3s)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/api/news")
async def get_news():
    return {"news": market_state["news"]}

# Startup event
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(update_system_state())

if __name__ == "__main__":
    import socket
    
    def is_port_in_use(port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('0.0.0.0', port)) == 0

    port = 8000
    while is_port_in_use(port):
        print(f"Port {port} is in use, trying {port+1}...")
        port += 1
        
    print(f"Starting server on port {port}...")
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=True)
