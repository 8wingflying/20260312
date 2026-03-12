
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
```
# 1. 載入資料集
```python
iris = load_iris()
# 將資料轉換為 Pandas DataFrame 方便操作
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
# 加入品種標籤
df['species'] = [iris.target_names[i] for i in iris.target]
```
```python
# 2. 基本統計描述 (集中趨勢與離散程度)
print("--- 總體描述性統計 ---")
print(df.describe())

# 3. 分組統計 (觀察不同品種的特性)
print("\n--- 各品種特徵平均值 ---")
grouped_mean = df.groupby('species').mean()
print(grouped_mean)

# 4. 相關係數矩陣 (查看特徵間的線性關係)
# 注意：計算相關係數前需移除非數值欄位
correlation = df.drop('species', axis=1).corr()
print("\n--- 特徵相關係數矩陣 ---")
print(correlation)

# --- 5. 資料視覺化 ---

# 設定繪圖風格
sns.set_theme(style="whitegrid")

# A. 盒鬚圖 (Boxplot)：觀察數值分佈與極端值
plt.figure(figsize=(12, 8))
for i, column in enumerate(iris.feature_names):
    plt.subplot(2, 2, i + 1)
    sns.boxplot(x='species', y=column, data=df, palette="Set2")
    plt.title(f'Distribution of {column}')
plt.tight_layout()
plt.show()

# B. 配對圖 (Pairplot)：一次看齊所有特徵的兩兩關係與分佈
# 這是分析 Iris 資料集最直觀的方式
sns.pairplot(df, hue="species", diag_kind="kde", palette="husl")
plt.suptitle("Iris Dataset Pairplot", y=1.02)
plt.show()

# C. 熱力圖 (Heatmap)：視覺化特徵相關性
plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()
```
