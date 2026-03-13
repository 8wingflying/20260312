
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
