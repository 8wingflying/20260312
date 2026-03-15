# 分類問題
- https://gemini.google.com/share/5d4fdd315687
- https://github.com/amineoucherif/MLP_IrisDataset
```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 1. 載入資料
iris = load_iris()
X, y = iris.data, iris.target

# 2. 資料預處理：MLP 對數值縮放非常敏感，建議先標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. 切分訓練集與測試集 (80% 訓練, 20% 測試)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 4. 建立 MLP 模型
# hidden_layer_sizes=(10,) 表示只有一層隱藏層，裡面有10個神經元
# max_iter=1000 表示最大迭代（學習）次數
mlp = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)

# 5. 訓練模型
mlp.fit(X_train, y_train)

# 6. 預測與評估
predictions = mlp.predict(X_test)
print(f"模型的準確度為: {accuracy_score(y_test, predictions) * 100:.2f}%")
```
# regression
- https://github.com/FirazKhan/California-Housing-Price-Prediction

```python
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 載入數據集
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = housing.target

# 分割訓練集與測試集 (80% 訓練, 20% 測試)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 標準化數據 (使特徵符合平均值 0, 方差 1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 初始化 MLP 回歸模型
mlp = MLPRegressor(
    hidden_layer_sizes=(100, 100), # 兩個隱藏層，各 100 個神經元
    activation='relu',             # 激活函數
    solver='adam',                 # 優化器
    max_iter=500,                  # 最大迭代次數
    random_state=42
)

# 訓練模型
mlp.fit(X_train_scaled, y_train)

# 進行預測
y_pred = mlp.predict(X_test_scaled)

# 評估指標
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"均方誤差 (MSE): {mse:.4f}")
print(f"R² 決定係數: {r2:.4f}")
```
#### 調整超參數（Hyperparameter Tuning）
- 針對 MLP 進行超參數調整（Hyperparameter Tuning）是提升模型性能的關鍵。
- 在 scikit-learn 中，最常用的方法是 GridSearchCV（網格搜索）或 RandomizedSearchCV（隨機搜索）。
```python
from sklearn.model_selection import GridSearchCV

# 定義參數網格
param_grid = {
    'hidden_layer_sizes': [(50, 50), (100,), (100, 50)],
    'activation': ['relu', 'tanh'],
    'alpha': [0.0001, 0.05],
    'learning_rate_init': [0.001, 0.01],
}

# 建立 MLP 模型
mlp = MLPRegressor(max_iter=500, random_state=42)

# 執行網格搜索 (cv=3 代表三折交叉驗證)
grid_search = GridSearchCV(mlp, param_grid, n_jobs=-1, cv=3, scoring='r2')
grid_search.fit(X_train_scaled, y_train)

# 輸出最佳參數
print(f"最佳參數組合: {grid_search.best_params_}")
print(f"最佳 R² 分數: {grid_search.best_score_:.4f}")

# 視覺化學習曲線 (Learning Curve)
import matplotlib.pyplot as plt

plt.plot(grid_search.best_estimator_.loss_curve_)
plt.title("Loss Curve (Best Model)")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.show()
```
