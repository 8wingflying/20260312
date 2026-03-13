# 探索性資料分析
- 針對鐵達尼號資料集（Titanic Dataset）進行 EDA（探索性資料分析，Exploratory Data Analysis） 是機器學習建模前最關鍵的一步。
- 這能幫助我們理解資料的分佈、找出異常值，並確認哪些特徵對「生存率」影響最大。

#### 針對該資料集的典型 EDA 步驟與觀察重點：
- `1`. 資料概覽與缺失值檢查
  - 首先要確認資料的完整度。鐵達尼號資料集通常會有嚴重的缺失值問題。
  - Age（年齡）： 約 20% 缺失。
  - Cabin（船艙號碼）： 約 77% 缺失（通常會直接捨棄或簡化處理）。
  - Embarked（登船港口）： 僅 2 筆缺失。
  - EDA 技巧： 使用 df.info() 或 sns.heatmap(df.isnull()) 來視覺化缺失值的比例。
- `2`. 單變數分析 (Univariate Analysis)
  - 觀察單一特徵的分佈情況。
  - 生存比例： 訓練集中大約只有 38% 的人倖存（這是一個類別不平衡問題，但並不極端）。
  - 艙等分佈： 三等艙（Pclass 3）的人數最多，超過總人數的一半。
  - 性別分佈： 男性乘客多於女性（約 65% vs 35%）。
- `3`. 雙變數分析 (Bivariate Analysis)：誰更容易生存？
  - 這是 EDA 的核心，目的是找出與 Survived 相關聯的特徵。
  - A. 性別與生存率 (Sex vs Survived)
    - 這是最強大的預測指標。
    - 觀察： 女性生存率高達 74% 左右，而男性僅約 18%。
    - 結論： 性別是模型中最重要的特徵。
  - B. 艙等與生存率 (Pclass vs Survived)
    - 社會經濟地位對生存率有顯著影響。
    - 觀察： 一等艙（Pclass 1）生存率最高（>60%），三等艙最低（<25%）。
    - 結論： 雖然三等艙人數最多，但獲救比例最低。
  - C. 年齡與生存率 (Age vs Survived)
    - 觀察： 幼兒（0-5歲）的生存率顯著較高。老年人的生存率則相對較低。
    - 結論： 年齡分佈呈現出「婦孺優先」的救援邏輯。
- `4`. 多變數分析與相關性 (Multivariate Analysis)
  - 透過相關係數矩陣（Heatmap）觀察變數間的線性關係。
  - Fare 與 Pclass： 強烈負相關（票價越高，Pclass 數字越小，即等級越高）。
  - SibSp 與 Parch： 有正相關，通常可以合併為 FamilySize（家庭成員總數）。
- `5`. 特徵工程的啟發 (Feature Engineering Insights)
  - 透過 EDA，我們可以得到以下特徵優化的靈感：
  - Name 稱謂： 提取出 Mr, Mrs, Miss, Master。例如，Master 通常代表年幼男孩，其生存率高於一般的 Mr。
  - 孤身一人 (IsAlone)： 如果 SibSp + Parch == 0，則標記為獨自乘船。數據顯示，獨自乘船者的死亡率通常較高。
  - 票價分箱 (Fare Binning)： 票價分佈極度偏態（少數人付了極高額票價），將其分段處理有助於模型收斂。

## 程式
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. 載入資料 (假設檔案名為 titanic.csv)
# 或者直接從 seaborn 載入內建資料集
df = sns.load_dataset('titanic')

# 設定繪圖風格
sns.set_theme(style="whitegrid")

# --- 圖表 A：缺失值視覺化 ---
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.title('Missing Data Heatmap (Yellow shows Missing)')
plt.show()

# --- 圖表 B：生存與性別的關係 ---
plt.figure(figsize=(8, 5))
sns.countplot(x='survived', hue='sex', data=df, palette='RdBu_r')
plt.title('Survival Count by Sex')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.show()

# --- 圖表 C：生存與艙等 (Pclass) 的關係 ---
plt.figure(figsize=(8, 5))
sns.countplot(x='survived', hue='pclass', data=df, palette='rainbow')
plt.title('Survival Count by Passenger Class')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.show()

# --- 圖表 D：年齡分佈與生存率 ---
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='age', hue='survived', kde=True, element="step", palette='magma')
plt.title('Age Distribution by Survival Status')
plt.xlabel('Age')
plt.show()

# --- 圖表 E：相關係數矩陣 (Heatmap) ---
plt.figure(figsize=(10, 8))
# 僅篩選數值型欄位進行運算
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Features')
plt.tight_layout()
plt.show()
```
