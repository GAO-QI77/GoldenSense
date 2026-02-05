import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import asyncio
import json
import random
import yfinance as yf
import pandas as pd
from datetime import datetime

app = FastAPI()

# 1. CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Static Files
app.mount("/static", StaticFiles(directory="live_platform/static"), name="static")

# Global Cache
market_cache = {
    "price": 2350.0,
    "timestamp": datetime.now().isoformat(),
    "history": []
}

async def fetch_market_data():
    """Background task to fetch real market data periodically"""
    while True:
        try:
            ticker = yf.Ticker("GC=F")
            data = ticker.history(period="1d", interval="1m")
            if not data.empty:
                last_price = float(data['Close'].iloc[-1])
                market_cache["price"] = last_price
                market_cache["timestamp"] = datetime.now().isoformat()
                # Keep last 50 points for chart initialization
                market_cache["history"] = data['Close'].tail(50).tolist()
        except Exception as e:
            print(f"Error fetching data: {e}")
        await asyncio.sleep(60) # Fetch every minute

# 3. Routes
@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("live_platform/static/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/api/history")
async def get_history():
    return {"history": market_cache["history"], "current": market_cache["price"]}

@app.get("/api/stream")
async def message_stream(request: Request):
    """Server-Sent Events (SSE) for real-time price updates"""
    async def event_generator():
        while True:
            # Simulate real-time fluctuation between fetches
            if await request.is_disconnected():
                break
            
            # Add small random noise to make it look "alive" between 1-min fetches
            noise = random.uniform(-0.5, 0.5)
            current_price = market_cache["price"] + noise
            
            data = json.dumps({
                "price": round(current_price, 2),
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "change_percent": random.uniform(-0.05, 0.05) # Simulated instant change
            })
            yield f"data: {data}\n\n"
            await asyncio.sleep(1) # Update every second

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/api/news")
async def get_news():
    """Returns latest news (Simulated for demo stability)"""
    news_items = [
        {"time": "Just now", "title": "Fed Chair Powell signals rate pause", "category": "Policy"},
        {"time": "2 mins ago", "title": "Gold breaks resistance at $2,400", "category": "Market"},
        {"time": "15 mins ago", "title": "ECB considers early rate cuts", "category": "Policy"},
        {"time": "1 hour ago", "title": "Geopolitical tensions rise in Middle East", "category": "Geopolitics"},
        {"time": "2 hours ago", "title": "Central Banks increase gold reserves", "category": "Data"},
    ]
    # Shuffle to simulate updates
    random.shuffle(news_items)
    return {"news": news_items[:3]}

# Startup event
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(fetch_market_data())

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
