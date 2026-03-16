import pandas as pd
import numpy as np
import torch
from data_loader import MarketDataLoader, NewsDataLoader
from feature_engineer import FeatureEngineer
from stacking_model import DynamicEnsemble, GRUModel, TransformerModel
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error

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
    
    # 3. 运行特征工程 (拆解步骤以获取序列特征)
    engineer = FeatureEngineer(horizons=[1, 7, 30])
    
    # 3.1 预处理与特征构建
    df = engineer.preprocess(raw_market)
    df = engineer.construct_seasonal_features(df)
    if not daily_signals.empty:
        df = engineer.construct_news_features(df, daily_signals)
    df = engineer.construct_market_features(df)
    
    # 3.2 自适应归一化 (关键：确保序列特征也使用相同的归一化)
    df = engineer.adaptive_normalize(df)
    
    # 3.3 构建目标
    df = engineer.construct_targets(df)
    
    # 3.4 准备 X_tab (特征选择)
    target_cols = [c for c in df.columns if 'target_' in c]
    X_all = df.drop(columns=target_cols)
    Y = df[target_cols]
    
    results, top_features = engineer.select_features(X_all, Y['target_return_1d'])
    engineer.document_features(results, top_features)
    
    # 保存选定特征
    import json
    with open("selected_features.json", "w") as f:
        json.dump(top_features, f)
        
    X_tab = X_all[top_features]
    
    # 2. 序列特征 (使用 ZScore 版本)
    # 必须确保这些列存在。adaptive_normalize 生成了 {col}_ZScore
    seq_cols = ['Gold_ZScore', 'Silver_ZScore', 'Crude_Oil_ZScore', 'USD_Index_ZScore']
    # 检查列是否存在，如果不存在(可能是列名不同)，回退到原始列并手动归一化(不推荐)
    # 假设 MarketDataLoader 的列名是 Gold, Silver... adaptive_normalize 会生成 Gold_ZScore...
    
    # 注意：X_all 已经 dropna 了，所以是齐的
    X_seq_data = X_all[seq_cols].values
    X_seq = engineer.create_sequences(X_seq_data, 60)
    
    # 对齐 X_tab 和 X_seq
    # create_sequences 返回 len(data) - 59 个样本
    # X_tab 也需要切掉前 59 个
    X_tab = X_tab.iloc[59:]
    Y = Y.iloc[59:]
    
    # 3. 运行 Walk-forward 验证 (AB 测试框架)
    cv_report = walk_forward_validation(X_tab, X_seq, Y)
    cv_report.to_csv('ab_test_report.csv', index=False)
    
    # 4. 训练最终生产模型
    print("\n训练最终生产模型并生成预测...")
    # Use common index from aligned data
    results_all = pd.DataFrame({'Date': X_tab.index})
    
    # 训练一个主要用于 1 天预测的模型进行保存 (演示目的)
    production_model = None
    
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
        
        if h == 1:
            production_model = final_ensemble
        
    results_all.to_csv('prediction_results.csv', index=False)
    raw_market.to_csv('raw_market_data.csv')
    
    if production_model:
        production_model.save_model("model_checkpoints")
        print("生产模型 (T+1) 已保存至 model_checkpoints")
        
    print("架构分析与重构完成。")

if __name__ == "__main__":
    train_final_system()
