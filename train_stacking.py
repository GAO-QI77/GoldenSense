import pandas as pd
import numpy as np
import torch
from data_loader import MarketDataLoader, NewsDataLoader
from feature_engineer import FeatureEngineer
from stacking_model import DynamicEnsemble, GRUModel, TransformerModel
from sklearn.model_selection import TimeSeriesSplit

def evaluate(y_true, y_pred, name="Model"):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    acc = ((y_pred > 0) == (y_true > 0)).mean()
    print(f"{name} - RMSE: {rmse:.6f}, MAE: {mae:.6f}, ACC: {acc:.2%}")
    return rmse, mae, acc

def walk_forward_validation(X_tab, X_seq, Y, horizons=[1, 7]):
    """
    实施 Walk-forward Validation 与 A/B 测试框架。
    """
    tscv = TimeSeriesSplit(n_splits=5)
    cv_metrics = []
    
    for h in horizons:
        print(f"\n===== 开始 T+{h} 周期 Walk-forward 验证 =====")
        y_h = Y[f'target_return_{h}d']
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X_tab)):
            X_t, X_v = X_tab.iloc[train_idx], X_tab.iloc[test_idx]
            Xs_t, Xs_v = X_seq[train_idx], X_seq[test_idx]
            y_t, y_v = y_h.iloc[train_idx], y_h.iloc[test_idx]
            
            # 1. 训练集成模型
            ensemble = DynamicEnsemble(tabular_input_dim=X_tab.shape[1], seq_input_dim=X_seq.shape[2])
            ensemble.train_l1(X_t, Xs_t, y_t, X_v, Xs_v, y_v)
            
            # 2. 自适应更新：使用最近 7 天数据更新权重
            ensemble.update_dynamic_weights(X_t.tail(7), Xs_t[-7:], y_t.tail(7))
            
            # 3. 预测与评估
            preds = ensemble.predict(X_v, Xs_v)
            rmse, mae, acc = evaluate(y_v, preds, f"Fold {fold+1}")
            cv_metrics.append({'Horizon': h, 'Fold': fold, 'Accuracy': acc, 'RMSE': rmse, 'MAE': mae})
            
    return pd.DataFrame(cv_metrics)

def train_final_system():
    # 1. 数据准备
    print("准备重构后的系统数据...")
    m_loader = MarketDataLoader()
    n_loader = NewsDataLoader()
    raw_market = m_loader.fetch_data(period='5y')
    news = n_loader.fetch_news()
    scored_news = n_loader.analyze_causality(news)
    daily_signals = n_loader.get_daily_signals(scored_news)
    
    engineer = FeatureEngineer(horizons=[1, 7, 30])
    X_tab, Y = engineer.run_pipeline(raw_market, daily_signals)
    
    # 2. 序列特征
    price_cols = ['Gold', 'Silver', 'Crude_Oil', 'USD_Index']
    price_scaled = (raw_market[price_cols].ffill().dropna() - raw_market[price_cols].mean()) / raw_market[price_cols].std()
    X_seq_all = engineer.create_sequences(price_scaled.values, 60)
    
    # 对齐
    common_index = X_tab.index.intersection(price_scaled.index[59:])
    X_tab, Y = X_tab.loc[common_index], Y.loc[common_index]
    seq_indices = [price_scaled.index.get_loc(idx) - 59 for idx in common_index]
    X_seq = X_seq_all[seq_indices]
    
    # 3. 运行 Walk-forward 验证 (AB 测试框架)
    cv_report = walk_forward_validation(X_tab, X_seq, Y)
    cv_report.to_csv('ab_test_report.csv', index=False)
    
    # 4. 训练最终生产模型
    print("\n训练最终生产模型并生成预测...")
    results_all = pd.DataFrame({'Date': common_index})
    
    for h in [1, 7, 30]:
        y_h = Y[f'target_return_{h}d']
        final_ensemble = DynamicEnsemble(tabular_input_dim=X_tab.shape[1], seq_input_dim=X_seq.shape[2])
        # 使用 90% 数据训练，10% 数据作为动态权重调整参考
        split = int(len(X_tab) * 0.9)
        final_ensemble.train_l1(X_tab[:split], X_seq[:split], y_h[:split], X_tab[split:], X_seq[split:], y_h[split:])
        final_ensemble.update_dynamic_weights(X_tab[split:], X_seq[split:], y_h[split:])
        
        preds = final_ensemble.predict(X_tab, X_seq)
        results_all[f'True_{h}d'] = y_h.values
        results_all[f'Pred_{h}d'] = preds
        results_all[f'Prob_{h}d'] = 0.5 + 0.4 * np.tanh(preds / y_h.std())
        
    results_all.to_csv('prediction_results.csv', index=False)
    raw_market.to_csv('raw_market_data.csv')
    print("架构分析与重构完成。")

if __name__ == "__main__":
    train_final_system()
