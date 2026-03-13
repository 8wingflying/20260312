## Holdout(保留)交叉驗證(train_test_split)
```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Load the iris dataset
iris = load_iris()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3)

# Train a k-nearest neighbors classifier on the training set
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Evaluate the performance of the classifier on the testing set
score = knn.score(X_test, y_test)
print('Accuracy:', score)
```
## 其他交叉驗證
```python
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold, LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# 1. 準備資料與模型
data = load_iris()
X, y = data.data, data.target
model = RandomForestClassifier(n_estimators=100, random_state=42)

# --- 方法 A: 標準 K-Fold (5折) ---
kf = KFold(n_splits=5, shuffle=True, random_state=42)
kf_scores = cross_val_score(model, X, y, cv=kf)

print(f"K-Fold 平均準確率: {kf_scores.mean():.4f}")
print(f"每折分數: {kf_scores}")

# --- 方法 B: 分層 Stratified K-Fold (處理類別不平衡) ---
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
skf_scores = cross_val_score(model, X, y, cv=skf)

print(f"\nStratified K-Fold 平均準確率: {skf_scores.mean():.4f}")

# --- 方法 C: 留一法 LOOCV (適合極小資料集) ---
loo = LeaveOneOut()
# 注意：LOOCV 會跑 N 次，資料量大時請慎用
loo_scores = cross_val_score(model, X, y, cv=loo)

print(f"\nLOOCV 平均準確率: {loo_scores.mean():.4f}")
print(f"總共訓練次數: {len(loo_scores)}")
```

## Shuffle-Split 交叉驗證
```python
from sklearn.datasets import load_iris
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# 1. 載入鳶尾花資料集
iris = load_iris()
X, y = iris.data, iris.target

# 2. 定義 ShuffleSplit 策略
# n_splits: 迭代次數 (跑幾次訓練/測試)
# test_size: 每次迭代中測試集所佔的比例 (0.3 表示 30%)
# random_state: 確保結果可被重複驗證
rs = ShuffleSplit(n_splits=10, test_size=0.3, random_state=42)

# 3. 初始化模型
model = RandomForestClassifier(n_estimators=100, random_state=42)

# 4. 執行交叉驗證
scores = cross_val_score(model, X, y, cv=rs)

# 5. 輸出結果
print(f"--- Shuffle-Split 驗證結果 (10次迭代) ---")
print(f"每輪準確率: \n{scores}")
print(f"\n平均準確率: {scores.mean():.4f}")
print(f"標準差 (穩定度): {scores.std():.4f}")

# 額外補充：如何手動查看索引 (Indices)
# 這能幫你理解它是如何隨機抽取的
train_index, test_index = next(rs.split(X))
print(f"\n第一輪的測試集索引範例: \n{test_index}")
```
