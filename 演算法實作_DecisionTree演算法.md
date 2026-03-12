# 演算法實作_DecisionTree演算法
- 從零開始（From Scratch）、不依賴 sklearn 等機器學習庫的決策樹實作。
- 為了保持「最簡單」的設計，這個類別將專注於 二元分類，並使用 資訊熵 (Entropy) 作為分割標準。

## 簡易決策樹 Python 實作
```python
import numpy as np
from collections import Counter

class SimpleDecisionTree:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.tree = None

    def _entropy(self, y):
        # 計算資訊熵：描述資料的混亂程度
        counts = np.bincount(y)
        probabilities = counts / len(y)
        return -np.sum([p * np.log2(p) for p in probabilities if p > 0])

    def _best_split(self, X, y):
        # 尋找最佳分割點：嘗試每個特徵與每個數值
        best_gain = -1
        split_idx, split_thresh = None, None
        
        parent_entropy = self._entropy(y)
        n_features = X.shape[1]

        for i in range(n_features):
            thresholds = np.unique(X[:, i])
            for t in thresholds:
                left_indices = np.where(X[:, i] <= t)[0]
                right_indices = np.where(X[:, i] > t)[0]
                
                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue

                # 計算加權後的子節點熵
                n = len(y)
                e_left, e_right = self._entropy(y[left_indices]), self._entropy(y[right_indices])
                child_entropy = (len(left_indices) / n) * e_left + (len(right_indices) / n) * e_right
                
                # 資訊增益 (Information Gain)
                ig = parent_entropy - child_entropy
                if ig > best_gain:
                    best_gain, split_idx, split_thresh = ig, i, t
        
        return split_idx, split_thresh

    def _build_tree(self, X, y, depth=0):
        # 終止條件：標籤純淨、達到最大深度、或無法再分
        n_labels = len(np.unique(y))
        if depth >= self.max_depth or n_labels == 1:
            return Counter(y).most_common(1)[0][0]

        idx, thresh = self._best_split(X, y)
        if idx is None: return Counter(y).most_common(1)[0][0]

        left_idx = np.where(X[:, idx] <= thresh)[0]
        right_idx = np.where(X[:, idx] > thresh)[0]

        return {
            'feature': idx, 'threshold': thresh,
            'left': self._build_tree(X[left_idx], y[left_idx], depth + 1),
            'right': self._build_tree(X[right_idx], y[right_idx], depth + 1)
        }

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _predict_one(self, x, tree):
        if not isinstance(tree, dict): return tree
        if x[tree['feature']] <= tree['threshold']:
            return self._predict_one(x, tree['left'])
        return self._predict_one(x, tree['right'])

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])

# --- 測試範例 ---
X = np.array([[2, 3], [10, 11], [1, 1], [12, 12]]) # 特徵：小數值 vs 大數值
y = np.array([0, 1, 0, 1])                       # 標籤：0 或 1

model = SimpleDecisionTree(max_depth=2)
model.fit(X, y)
print(f"預測結果: {model.predict(np.array([[2, 2], [15, 15]]))}")
```
