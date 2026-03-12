# 演算法實作_KNN演算法
- KNN（K-Nearest Neighbors，K-近鄰演算法）的邏輯比決策樹更直覺：「物以類聚」。
- 它不需要複雜的訓練過程，只需在預測時計算新樣本與所有已知樣本的距離。

## 使用 numpy 實作的最簡約版本
```python
import numpy as np
from collections import Counter
```
```python
class SimpleKNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        # KNN 屬於「懶惰學習」，fit 階段只需儲存資料
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def _euclidean_distance(self, x1, x2):
        # 計算歐幾里得距離 (L2 Norm)
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def _predict_one(self, x):
        # 1. 計算新樣本與所有訓練樣本的距離
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
        
        # 2. 取得距離最近的 K 個索引
        k_indices = np.argsort(distances)[:self.k]
        
        # 3. 提取這些索引對應的標籤
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # 4. 多數決 (Majority Vote)
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def predict(self, X):
        X = np.array(X)
        return np.array([self._predict_one(x) for x in X])
```
```python
# --- 測試範例 ---
# 特徵：[身高, 體重]，標籤：0 (貓), 1 (狗)
X_train = [[35, 5], [40, 7], [80, 25], [85, 30]]
y_train = [0, 0, 1, 1]

model = SimpleKNN(k=3)
model.fit(X_train, y_train)

# 測試一個體型中等的動物 [50, 12]
test_data = [[50, 12]]
prediction = model.predict(test_data)
print(f"預測標籤: {prediction[0]} (0 是貓, 1 是狗)")
```

### 修正版
- 在 KNN 演算法中，如果特徵的「單位」或「量級」落差太大（例如：身高 180 cm vs. 體重 0.07 公噸），數值大的特徵會完全主導距離計算，導致模型忽略掉重要的細小特徵。
- 為了修正這個問題，我們通常會加入 Z-分數標準化 (Z-score Standardization) ==>具備特徵縮放功能的 KNN Class
- 在 SimpleKNN 中加入 StandardScaler 的邏輯，確保每個特徵的平均值為 0，標準差為 1。

```python
import numpy as np
from collections import Counter

class ScaledKNN:
    def __init__(self, k=3):
        self.k = k
        self.mean = None
        self.std = None

    def _scale(self, X, training=False):
        # 如果是訓練階段，計算並紀錄平均值與標準差
        if training:
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0)
        
        # 執行標準化：(x - μ) / σ
        # 避免除以 0 的情況發生
        return (X - self.mean) / (self.std + 1e-8)

    def fit(self, X, y):
        X = np.array(X)
        # 1. 訓練時先進行縮放，並記住縮放參數
        self.X_train = self._scale(X, training=True)
        self.y_train = np.array(y)

    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def _predict_one(self, x):
        # 2. 預測點也必須使用「訓練集」的參數進行縮放
        x_scaled = (x - self.mean) / (self.std + 1e-8)
        
        distances = [self._euclidean_distance(x_scaled, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        return Counter(k_nearest_labels).most_common(1)[0][0]

    def predict(self, X):
        return np.array([self._predict_one(x) for x in X])

# --- 實驗對照 ---
# 特徵 1 (身高 150~190), 特徵 2 (體重 0.05~0.1 公噸) -> 量級差異極大
X_raw = np.array([[180, 0.08], [160, 0.05], [170, 0.07], [190, 0.09]])
y_train = [1, 0, 0, 1]

model = ScaledKNN(k=3)
model.fit(X_raw, y_train)

# 測試新資料
X_test = np.array([[175, 0.075]])
print(f"縮放後的預測結果: {model.predict(X_test)}")
```
