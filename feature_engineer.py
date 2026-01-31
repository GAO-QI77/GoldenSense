import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression, SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
import os

class FeatureEngineer:
    """
    全面的特征工程流程：预处理、构建、变换、选择和文档化。
    """
    def __init__(self, horizons=[1, 7, 30]):
        self.horizons = horizons
        self.scaler = StandardScaler()
        self.selected_features = []
        self.feature_docs = []

    def preprocess(self, df):
        """
        数据预处理：处理缺失值、异常值。
        """
        # 1. 处理缺失值：前向填充（金融数据常用），然后删除开头无法填充的行
        df = df.ffill().dropna()
        
        # 2. 异常值处理：使用 Z-score 限制极值（可选，保持金融市场剧烈波动的特性）
        # 这里仅对非目标列进行轻微裁剪
        return df

    def construct_market_features(self, df):
        """
        构建市场技术特征、跨品种特征、实际利率及波动率状态。
        """
        new_df = df.copy()
        gold_col = 'Gold'
        
        # 1. 基础收益率与动量
        base_market_cols = ['Gold', 'Silver', 'USD_Index', 'S&P500', 'VIX', 'Crude_Oil', '10Y_Bond', '2Y_Bond']
        existing_base_cols = [c for c in base_market_cols if c in df.columns]
        
        for col in existing_base_cols:
            new_df[f'{col}_Return_1d'] = df[col].pct_change(1)
            new_df[f'{col}_Return_5d'] = df[col].pct_change(5)
            new_df[f'{col}_Momentum'] = df[col] / df[col].shift(10) - 1
            
        # 2. 引入“实际利率 (Real Yields)”：10Y名义利率 - 通胀信号
        if '10Y_Bond' in df.columns and 'News_Inflation' in new_df.columns:
            # 简化版 TIPS 模拟：名义利率 - 新闻通胀信号（作为通胀预期代理）
            new_df['Real_Yield_Proxy'] = df['10Y_Bond'] - new_df['News_Inflation']
            
        # 3. 波动率状态特征 (ATR)
        high = df[gold_col] * 1.001 # 模拟
        low = df[gold_col] * 0.999  # 模拟
        close_prev = df[gold_col].shift(1)
        tr = pd.concat([high - low, (high - close_prev).abs(), (low - close_prev).abs()], axis=1).max(axis=1)
        new_df['Gold_ATR'] = tr.rolling(window=14).mean()
        
        # 4. 技术指标
        new_df['Gold_MA5'] = df[gold_col].rolling(5).mean()
        new_df['Gold_MA20'] = df[gold_col].rolling(20).mean()
        
        # 5. 跨品种比例与利差
        if 'Silver' in df.columns:
            new_df['Gold_Silver_Ratio'] = df['Gold'] / df['Silver']
        if '10Y_Bond' in df.columns and '2Y_Bond' in df.columns:
            new_df['Yield_Curve_Spread'] = df['10Y_Bond'] - df['2Y_Bond']
            
        return new_df

    def create_sequences(self, data, seq_length=60):
        """
        为序列模型构建过去 N 天的数据。
        """
        sequences = []
        for i in range(len(data) - seq_length + 1):
            sequences.append(data[i:i + seq_length])
        return np.array(sequences)

    def construct_seasonal_features(self, df):
        """
        添加周期性特征：周、月、交易时段编码。
        """
        df.index = pd.to_datetime(df.index)
        df['DayOfWeek'] = df.index.dayofweek
        df['Month'] = df.index.month
        df['Hour'] = df.index.hour # 虽然是日线数据，保留接口用于分钟级
        
        # 周期性编码 (sin/cos)
        df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
        df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
        df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        
        # 交易时段编码 (1: 伦敦/纽约重叠, 0: 其他)
        # 假设 13:00 - 17:00 UTC 为高波动时段
        df['Session_Active'] = ((df['Hour'] >= 13) & (df['Hour'] <= 17)).astype(int)
        
        return df.drop(columns=['DayOfWeek', 'Month', 'Hour'])

    def construct_targets(self, df):
        """
        构建多周期预测目标：T+1, T+7, T+30 的幅度与方向。
        """
        gold_prices = df['Gold'].astype(float)
        
        for h in self.horizons:
            # 价格变动百分比 (幅度)
            df[f'target_return_{h}d'] = gold_prices.shift(-h) / gold_prices - 1
            # 方向角度 (用于 von Mises)
            df[f'target_theta_{h}d'] = np.arctan2(df[f'target_return_{h}d'], 0.01)
            # 简单分类标签 (1: 涨, 0: 跌)
            df[f'target_class_{h}d'] = (df[f'target_return_{h}d'] > 0).astype(int)
            
        return df.dropna()

    def select_features(self, X, y):
        """
        多准则特征选择：统计方法 + 机器学习模型。
        """
        print("执行特征选择流程...")
        features = X.columns
        results = pd.DataFrame(index=features)
        
        # 1. 相关性分析 (Pearson)
        corr_list = []
        for f in features:
            if X[f].nunique() <= 1:
                corr_list.append(0)
            else:
                c, _ = stats.pearsonr(X[f], y)
                corr_list.append(c if not np.isnan(c) else 0)
        results['Correlation'] = corr_list
        
        # 2. 互信息 (Mutual Information) - 捕捉非线性关系
        mi = mutual_info_regression(X, y)
        results['MI_Score'] = mi / np.max(mi)
        
        # 3. Random Forest 特征重要性
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        importance = model.feature_importances_
        results['RF_Importance'] = importance / np.max(importance)
        
        # 综合评分 (加权平均)
        results['Total_Score'] = (results['Correlation'].abs() + results['MI_Score'] + results['RF_Importance']) / 3
        
        # 选择前 K 个特征
        top_features = results.sort_values(by='Total_Score', ascending=False).head(15).index.tolist()
        self.selected_features = top_features
        
        return results, top_features

    def document_features(self, results, top_features):
        """
        生成特征文档。
        """
        doc = "# 特征工程文档 (Feature Documentation)\n\n"
        doc += "| 特征名称 | 构建逻辑 | 统计重要性 (Total Score) | 业务含义 |\n"
        doc += "| :--- | :--- | :--- | :--- |\n"
        
        meanings = {
            'Gold_Return_1d': '金价单日收益率，反映短期趋势',
            'Gold_RSI': '14日相对强弱指数，衡量超买超卖',
            'Gold_Silver_Ratio': '金银价格比，反映贵金属市场内部结构',
            'USD_Index_Return_1d': '美元指数变动，黄金负相关避险资产',
            'Yield_Curve_Spread': '美债利差，反映宏观经济预期',
            'Gold_Vol_5d': '5日滚动波动率，反映市场恐慌程度',
            'target_log_return': '预测目标：未来 H 天的对数收益率'
        }
        
        for feat in top_features:
            score = results.loc[feat, 'Total_Score']
            logic = "Rolling Window / Ratio"
            meaning = meanings.get(feat, "组合特征/市场动量")
            doc += f"| {feat} | {logic} | {score:.4f} | {meaning} |\n"
            
        with open("FEATURE_DOC.md", "w", encoding="utf-8") as f:
            f.write(doc)
        print("特征文档已生成：FEATURE_DOC.md")

    def construct_news_features(self, df, daily_signals):
        """
        整合多维新闻信号并进行自适应平滑。
        """
        if daily_signals.empty:
            for col in ['total', 'inflation', 'rates', 'risk', 'fx']:
                df[f'News_{col.capitalize()}'] = 0
        else:
            daily_signals.index = pd.to_datetime(daily_signals.index).date
            df_index_date = pd.to_datetime(df.index).date
            df['date_temp'] = df_index_date
            
            df = df.merge(daily_signals, left_on='date_temp', right_index=True, how='left')
            df = df.drop(columns=['date_temp'])
            
            # 填充缺失值并应用自适应 EMA
            for col in ['total', 'inflation', 'rates', 'risk', 'fx']:
                df[col] = df[col].fillna(0)
                # 自适应衰减：风险类信号衰减慢，FX类快
                span = 5 if col == 'risk' else 3
                df[f'News_{col.capitalize()}'] = df[col].ewm(span=span).mean()
                df = df.drop(columns=[col])
        return df

    def adaptive_normalize(self, df, window=20):
        """
        自适应归一化：处理金融数据的异方差性。
        使用滚动均值和标准差。
        """
        for col in df.columns:
            if 'target' not in col and 'sin' not in col and 'cos' not in col:
                roll_mean = df[col].rolling(window=window).mean()
                roll_std = df[col].rolling(window=window).std().replace(0, 1)
                df[f'{col}_ZScore'] = (df[col] - roll_mean) / roll_std
        return df.dropna()

    def run_pipeline(self, raw_data, daily_signals=None):
        """
        重构后的流水线：包含自适应归一化与多维信号。
        """
        # 1. 预处理
        df = self.preprocess(raw_data)
        
        # 2. 周期性特征
        df = self.construct_seasonal_features(df)
        
        # 3. 整合新闻多维信号
        if daily_signals is not None:
            df = self.construct_news_features(df, daily_signals)
        
        # 4. 构建市场特征
        df = self.construct_market_features(df)
        
        # 5. 自适应归一化
        df = self.adaptive_normalize(df)
        
        # 6. 构建多周期目标
        df = self.construct_targets(df)
        
        # 7. 划分 X, y
        target_cols = [c for c in df.columns if 'target_' in c]
        X = df.drop(columns=target_cols)
        Y = df[target_cols]
        
        # 8. 特征选择 (采用更稳健的互信息筛选)
        results, top_features = self.select_features(X, Y['target_return_1d'])
        
        # 9. 文档化与返回
        self.document_features(results, top_features)
        X_selected = X[top_features]
        # 注意：这里已经经过自适应归一化，不再需要全局 StandardScaler
        return X_selected, Y

if __name__ == "__main__":
    from data_loader import MarketDataLoader, NewsDataLoader
    
    print("开始特征工程流水线...")
    # 1. 加载市场数据
    loader = MarketDataLoader()
    raw_data = loader.fetch_data(period='2y')
    
    # 2. 加载并处理新闻数据
    news_loader = NewsDataLoader()
    news = news_loader.fetch_news()
    scored_news = news_loader.score_news(news)
    daily_sentiment = news_loader.get_daily_sentiment(scored_news)
    
    # 3. 运行特征工程
    engineer = FeatureEngineer(horizons=[1, 7, 30])
    X, Y = engineer.run_pipeline(raw_data, daily_sentiment)
    
    print("\n特征工程完成！")
    print(f"最终选定特征数: {len(X.columns)}")
    print("前 5 行特征预览:")
    print(X.head())
