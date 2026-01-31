import yfinance as yf
import pandas as pd
import numpy as np
import feedparser
from bs4 import BeautifulSoup
import requests
from datetime import datetime, timedelta
import time
import re

class MarketDataLoader:
    """
    Fetches market data from yfinance.
    """
    def __init__(self, tickers=None):
        if tickers is None:
            # Default tickers for gold price influence
            self.tickers = {
                'Gold': 'GC=F',        # Gold Futures
                'Silver': 'SI=F',      # Silver Futures
                'USD_Index': 'DX=F',   # US Dollar Index
                'S&P500': '^GSPC',     # S&P 500
                'VIX': '^VIX',         # Volatility Index
                'Crude_Oil': 'CL=F',   # Crude Oil Futures
                '10Y_Bond': '^TNX',    # 10Y Treasury Note Yield
                '2Y_Bond': '^IRX'      # 13 Week Treasury Bill (Proxy for 2Y or short term)
            }
        else:
            self.tickers = tickers

    def fetch_data(self, period='2y', interval='1d'):
        """
        Fetches historical data for all tickers.
        """
        data_frames = {}
        for name, ticker in self.tickers.items():
            print(f"Fetching {name} ({ticker})...")
            df = yf.download(ticker, period=period, interval=interval)
            if not df.empty:
                # yfinance returns a DataFrame, sometimes with MultiIndex columns
                if isinstance(df.columns, pd.MultiIndex):
                    # Extract 'Close' from the first level if it exists
                    if 'Close' in df.columns.get_level_values(0):
                        data_frames[name] = df['Close'].iloc[:, 0]
                else:
                    data_frames[name] = df['Close']
            else:
                print(f"Warning: No data for {ticker}")
        
        # Combine all into one DataFrame
        combined_df = pd.DataFrame(data_frames)
        combined_df = combined_df.ffill().dropna()
        return combined_df

class NewsDataLoader:
    """
    高级新闻数据加载器：包含因果分析、情感量化与重要性加权。
    """
    def __init__(self, feeds=None):
        if feeds is None:
            self.feeds = [
                'https://www.reutersagency.com/feed/?best-topics=business-finance',
                'https://search.cnbc.com/rs/search/combined?partnerId=2&keywords=gold%20price',
                'https://www.investing.com/rss/news_95.rss',
                'https://www.fxstreet.com/rss/news'
            ]
        else:
            self.feeds = feeds
            
        # 扩展的因果字典：[情感极性, 影响类别, 强度]
        # 影响类别: 1: 通胀, 2: 利率, 3: 避险, 4: 汇率
        self.causal_lexicon = {
            'inflation': [1.5, 1, 0.8],
            'cpi': [1.2, 1, 0.7],
            'ppi': [1.0, 1, 0.6],
            'hyperinflation': [2.5, 1, 1.0],
            
            'rate cut': [1.8, 2, 0.9],
            'dovish': [1.4, 2, 0.7],
            'easing': [1.5, 2, 0.8],
            'rate hike': [-1.8, 2, 0.9],
            'hawkish': [-1.4, 2, 0.7],
            'tightening': [-1.5, 2, 0.8],
            
            'war': [2.5, 3, 1.0],
            'conflict': [2.0, 3, 0.9],
            'geopolitical': [1.8, 3, 0.8],
            'uncertainty': [1.2, 3, 0.5],
            'safe haven': [2.0, 3, 0.9],
            'crisis': [2.2, 3, 1.0],
            
            'weak dollar': [1.5, 4, 0.7],
            'dollar slips': [1.2, 4, 0.6],
            'strong dollar': [-1.5, 4, 0.7],
            'dollar surges': [-1.8, 4, 0.8],
            'gold demand': [1.6, 0, 0.7],
            'central bank buying': [1.8, 0, 0.8]
        }

    def fetch_news(self):
        news_items = []
        for url in self.feeds:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries:
                    # 提取更多元数据用于加权
                    news_items.append({
                        'title': entry.title,
                        'summary': getattr(entry, 'summary', ''),
                        'published': getattr(entry, 'published', datetime.now().strftime('%Y-%m-%d')),
                        'source': url.split('/')[2]
                    })
            except Exception as e:
                print(f"Error fetching {url}: {e}")
        return news_items

    def analyze_causality(self, news_items):
        """
        量化新闻的因果影响。
        """
        scored_items = []
        for item in news_items:
            text = (item['title'] + " " + item['summary']).lower()
            
            # 基础分数与类别分数
            scores = {'total': 0, 'inflation': 0, 'rates': 0, 'risk': 0, 'fx': 0}
            
            # 识别重要性加权 (头条新闻识别)
            importance = 1.2 if 'urgent' in text or 'breaking' in text else 1.0
            
            for word, (weight, category, strength) in self.causal_lexicon.items():
                if word in text:
                    val = weight * strength * importance
                    scores['total'] += val
                    if category == 1: scores['inflation'] += val
                    elif category == 2: scores['rates'] += val
                    elif category == 3: scores['risk'] += val
                    elif category == 4: scores['fx'] += val
            
            item.update(scores)
            scored_items.append(item)
        return scored_items

    def get_daily_signals(self, scored_items):
        """
        生成每日因果特征信号。
        """
        df = pd.DataFrame(scored_items)
        if df.empty: return pd.DataFrame()
        
        df['date'] = pd.to_datetime(df['published'], errors='coerce').dt.date
        df = df.dropna(subset=['date'])
        
        # 聚合多维度信号
        daily_signals = df.groupby('date').agg({
            'total': 'mean',
            'inflation': 'sum',
            'rates': 'sum',
            'risk': 'sum',
            'fx': 'sum'
        })
        return daily_signals

if __name__ == "__main__":
    # Test script
    print("Testing Market Data Loader...")
    market_loader = MarketDataLoader()
    market_data = market_loader.fetch_data(period='1mo')
    print("\nMarket Data (last 5 rows):")
    print(market_data.tail())

    print("\nTesting News Data Loader...")
    news_loader = NewsDataLoader()
    news = news_loader.fetch_news()
    scored_news = news_loader.score_news(news)
    print(f"\nFetched {len(scored_news)} news items.")
    
    if scored_news:
        print("\nTop Scored News:")
        scored_news.sort(key=lambda x: abs(x['score']), reverse=True)
        for i in range(min(3, len(scored_news))):
            print(f"- {scored_news[i]['title']} (Score: {scored_news[i]['score']})")
            
        daily_sentiment = news_loader.get_daily_sentiment(scored_news)
        print("\nDaily Sentiment Aggregate:")
        print(daily_sentiment)
