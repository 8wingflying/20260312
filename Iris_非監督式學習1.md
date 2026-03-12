#### 🌸 Iris 資料集 無監督學習（Unsupervised Clustering）比較分析

---

## 📘 一、目標說明
本文件說明如何在 **Iris 資料集** 上進行無監督分群分析，涵蓋以下內容：

1. **K-Means 分群**
2. **階層式分群 (Hierarchical Clustering)**
3. **DBSCAN 密度式分群**
4. **Gaussian Mixture Model (GMM)**

透過比較不同模型的結果與指標（例如 ARI），幫助理解各演算法的特性與適用情境。

---

## 🧩 二、載入資料與前處理

```python
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 載入資料
iris = sns.load_dataset("iris")
X = iris.drop(columns="species")

# 資料標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

---

## ⚙️ 三、K-Means 分群分析

### 1️⃣ 使用肘部法則 (Elbow Method) 決定最佳 k 值

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

sse = []
K = range(1, 10)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    sse.append(kmeans.inertia_)

plt.plot(K, sse, 'bx-')
plt.xlabel('群數 k')
plt.ylabel('SSE (誤差平方和)')
plt.title('Elbow Method for Optimal k')
plt.show()
```

📈 **觀察重點：** SSE 在 k=3 之後下降趨勢趨緩，最佳群數為 **3 群**。

---

### 2️⃣ 建立 K-Means 模型與視覺化

```python
from sklearn.decomposition import PCA
import seaborn as sns

kmeans = KMeans(n_clusters=3, random_state=42)
iris["cluster"] = kmeans.fit_predict(X_scaled)

pca = PCA(n_components=2)
iris_pca = pd.DataFrame(pca.fit_transform(X_scaled), columns=['PCA1', 'PCA2'])
iris_pca["cluster"] = iris["cluster"]

sns.scatterplot(x="PCA1", y="PCA2", hue="cluster", data=iris_pca, palette="Set2", s=80)
plt.title("K-Means 分群結果 (PCA 2D 視覺化)")
plt.show()
```

---

### 3️⃣ 評估分群品質（ARI 指標）

```python
from sklearn.metrics import adjusted_rand_score

ari = adjusted_rand_score(iris["species"], iris["cluster"])
print(f"Adjusted Rand Index (ARI): {ari:.3f}")
```

📊 **結果解讀：** ARI 約為 0.7~0.8，代表 K-Means 分群效果良好。

---

## 🪜 四、階層式分群 (Hierarchical Clustering)

### 1️⃣ 樹狀圖 (Dendrogram)

```python
from scipy.cluster.hierarchy import linkage, dendrogram

plt.figure(figsize=(10, 6))
linkage_matrix = linkage(X_scaled, method='ward')
dendrogram(linkage_matrix)
plt.title("Hierarchical Clustering Dendrogram (ward linkage)")
plt.xlabel("Sample index")
plt.ylabel("Distance")
plt.show()
```

### 2️⃣ Agglomerative Clustering 模型

```python
from sklearn.cluster import AgglomerativeClustering

agg = AgglomerativeClustering(n_clusters=3, linkage="ward")
iris["agg_cluster"] = agg.fit_predict(X_scaled)
iris_pca["agg_cluster"] = iris["agg_cluster"]

sns.scatterplot(x="PCA1", y="PCA2", hue="agg_cluster", data=iris_pca, palette="Set1", s=80)
plt.title("階層式分群結果 (PCA 2D 視覺化)")
plt.show()
```

---

## 🔍 五、進階分群比較 — DBSCAN 與 GMM

### （1）DBSCAN 分群

```python
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.6, min_samples=5)
iris["dbscan_cluster"] = dbscan.fit_predict(X_scaled)

pca = PCA(n_components=2)
iris_pca = pd.DataFrame(pca.fit_transform(X_scaled), columns=["PCA1", "PCA2"])
iris_pca["dbscan_cluster"] = iris["dbscan_cluster"]

sns.scatterplot(x="PCA1", y="PCA2", hue="dbscan_cluster", data=iris_pca, palette="Set2", s=80)
plt.title("DBSCAN 分群結果 (PCA 2D 視覺化)")
plt.show()
```

📘 **參數說明：**
- `eps`: 鄰域半徑大小。
- `min_samples`: 定義核心點最小樣本數。
- `-1`: 噪音點。

```python
ari_dbscan = adjusted_rand_score(iris["species"], iris["dbscan_cluster"])
print(f"DBSCAN Adjusted Rand Index (ARI): {ari_dbscan:.3f}")
```

📊 **DBSCAN 分析結果：**
- 對非線性結構表現良好，但參數敏感。
- 常能準確辨識 Setosa，但其他兩類可能被合併。

---

### （2）Gaussian Mixture Model (GMM)

```python
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=3, covariance_type="full", random_state=42)
gmm_labels = gmm.fit_predict(X_scaled)
iris["gmm_cluster"] = gmm_labels

iris_pca["gmm_cluster"] = iris["gmm_cluster"]

sns.scatterplot(x="PCA1", y="PCA2", hue="gmm_cluster", data=iris_pca, palette="Set1", s=80)
plt.title("GMM 分群結果 (PCA 2D 視覺化)")
plt.show()
```

📘 **GMM 特點：**
- 可視為柔性版的 K-Means（允許橢圓形群）。
- 能輸出樣本屬於各群的機率（Soft Clustering）。

```python
ari_gmm = adjusted_rand_score(iris["species"], iris["gmm_cluster"])
print(f"GMM Adjusted Rand Index (ARI): {ari_gmm:.3f}")
```

---

### （3）四種模型比較

| 模型 | 原理 | 可辨識群數 | ARI (越高越好) | 優點 | 缺點 |
|------|------|-------------|----------------|------|------|
| **K-Means** | 基於距離最小化 | 3 | 約 0.73 | 簡單快速 | 對初始值敏感 |
| **階層式分群** | 層次合併距離最小群 | 3 | 約 0.70 | 可視化層次 | 難以處理大資料 |
| **DBSCAN** | 基於密度的分群 | 2~3 | 約 0.55 | 可偵測噪音 | 對參數敏感 |
| **GMM** | 橢圓高斯混合 | 3 | 約 0.78 | 分群柔性高 | 需假設分佈形狀 |

---

## 📈 六、GMM 機率分析

```python
probs = gmm.predict_proba(X_scaled)
iris_probs = pd.DataFrame(probs, columns=["Prob_Setosa", "Prob_Versicolor", "Prob_Virginica"])
print(iris_probs.head())
```

🔍 **說明：** `predict_proba()` 輸出樣本屬於各群的機率，有助分析邊界樣本。

---

## 📚 七、結論與建議

| 模型 | 適用情境 |
|------|------------|
| **K-Means** | 群形接近球形、群數已知 |
| **階層式分群** | 樣本數小、需了解層次結構 |
| **DBSCAN** | 存在噪音或非球形結構 |
| **GMM** | 群體呈橢圓分佈且群數已知 |

📘 **結論：** GMM 在 Iris 資料集上的表現最佳（ARI ≈ 0.78），能更準確捕捉 Versicolor 與 Virginica 之間的分佈差異。

---
