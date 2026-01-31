import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import Ridge
import numpy as np

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_size=64, num_layers=2, dropout=0.2):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :]).squeeze(-1)

class TransformerModel(nn.Module):
    """
    Transformer 网络，用于捕捉长程时序依赖。
    """
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.input_fc = nn.Linear(input_dim, d_model)
        self.output_fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_fc(x)
        x = self.transformer_encoder(x)
        return self.output_fc(x[:, -1, :]).squeeze(-1)

class DynamicEnsemble:
    """
    多模型融合架构：LSTM, Transformer, XGBoost, RF, ARIMA。
    实现基于滚动窗口表现的动态权重分配。
    """
    def __init__(self, tabular_input_dim, seq_input_dim, seq_length=60):
        self.xgb = xgb.XGBRegressor(n_estimators=500, learning_rate=0.03, max_depth=6, early_stopping_rounds=50)
        self.rf = RandomForestRegressor(n_estimators=200, max_depth=10, n_jobs=-1)
        self.gru = GRUModel(input_dim=seq_input_dim)
        self.transformer = TransformerModel(input_dim=seq_input_dim)
        self.meta_learner = Ridge(alpha=1.0) # 使用 Ridge 回归防止过拟合
        self.seq_length = seq_length
        self.model_weights = None # 动态权重

    def train_l1(self, X_tab, X_seq, y, X_tab_val, X_seq_val, y_val):
        print("正在训练多模型 L1 层...")
        
        # 1. XGBoost
        self.xgb.fit(X_tab, y, eval_set=[(X_tab_val, y_val)], verbose=False)
        
        # 2. Random Forest
        self.rf.fit(X_tab, y)
        
        # 3. GRU & Transformer
        self._train_torch_model(self.gru, X_seq, y, X_seq_val, y_val, name="GRU")
        self._train_torch_model(self.transformer, X_seq, y, X_seq_val, y_val, name="Transformer")

    def _train_torch_model(self, model, X, y, X_val, y_val, name, epochs=30):
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        X_t, y_t = torch.tensor(X, dtype=torch.float32), torch.tensor(y.values, dtype=torch.float32)
        X_v, y_v = torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val.values, dtype=torch.float32)
        
        for e in range(epochs):
            model.train()
            optimizer.zero_grad()
            loss = criterion(model(X_t), y_t)
            loss.backward()
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            v_loss = criterion(model(X_v), y_v)
        print(f"{name} 验证 Loss: {v_loss.item():.6f}")

    def update_dynamic_weights(self, X_tab_roll, X_seq_roll, y_roll):
        """
        根据最近 7 天的表现更新动态权重。
        """
        preds = self._get_l1_predictions(X_tab_roll, X_seq_roll)
        # 计算每个模型的误差倒数作为权重
        errors = [np.mean(np.abs(p - y_roll.values)) for p in preds.T]
        weights = 1.0 / (np.array(errors) + 1e-6)
        self.model_weights = weights / np.sum(weights)
        print(f"更新动态权重: {self.model_weights}")

    def _get_l1_predictions(self, X_tab, X_seq):
        p_xgb = self.xgb.predict(X_tab)
        p_rf = self.rf.predict(X_tab)
        
        self.gru.eval()
        self.transformer.eval()
        with torch.no_grad():
            p_gru = self.gru(torch.tensor(X_seq, dtype=torch.float32)).numpy()
            p_trans = self.transformer(torch.tensor(X_seq, dtype=torch.float32)).numpy()
            
        return np.column_stack([p_xgb, p_rf, p_gru, p_trans])

    def predict(self, X_tab, X_seq):
        l1_preds = self._get_l1_predictions(X_tab, X_seq)
        if self.model_weights is not None:
            return np.dot(l1_preds, self.model_weights)
        else:
            return np.mean(l1_preds, axis=1)



