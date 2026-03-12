# Ensemble Learning（集成學習）
- 核心思想是「團結力量大」，透過組合多個`弱學習器（Weak Learners）`來構建一個`強學習器`，以達到降低誤差、提高模型穩定性的目的。
- 三個臭皮匠勝過一個朱葛亮
- 根據組合策略的不同，主要可以分為以下四大類：
  - Voting (投票法)
  - Bagging (Bootstrap Aggregating)
  - Boosting
  - Stacking (Stacked Generalization)
## 1. Voting (投票法)
- 核心原理： 最直觀的方法。針對同一個資料集訓練多個不同的模型，然後根據所有模型的預測結果進行投票。
- Hard Voting： 少數服從多數，看預測類別出現次數。
- Soft Voting： 考慮每個預測類別的機率值（權重平均），通常比 Hard Voting 更精準。
## 2. Bagging (Bootstrap Aggregating)
- 核心原理： 透過「自助抽樣法」（Bootstrap Sampling）從訓練集中隨機抽取多個子集，分別訓練多個獨立的決策樹
- 最後以**投票（分類）或平均（迴歸）**的方式整合結果。
- 這能有效降低模型的變異數（Variance），防止過擬合。
- 代表演算法：
  - Random Forest (隨機森林)： 最具代表性的演算法。除了樣本隨機抽樣，還在特徵選擇上引入隨機性。
  - Bagged Decision Trees： 單純使用 Bagging 架構的決策樹組合。

## 3. Boosting
- 核心原理： 一種序列式的學習方法。每一個新模型都會試圖糾正前一個模型的錯誤。
- 具體做法是增加被錯誤分類樣本的權重，讓後續的模型更專注於處理這些「難題」。
- 這主要用於降低偏差（Bias）。
- 代表演算法：
  - AdaBoost (Adaptive Boosting)： 經典的啟始演算法，動態調整樣本權重。
  - GBDT (Gradient Boosting Decision Tree)： 利用梯度下降來優化損失函數。
- 框架與套件
  - XGBoost / LightGBM / CatBoost： 目前競賽與工業界最主流的高效能梯度提升框架，在速度與準確率上做了極大優化。

## 4. Stacking (Stacked Generalization)
- 核心原理： 這種方法比前兩者更複雜。
- 它分為兩層：
  - 第一層 (Base-Models)： 使用多個不同的演算法（如邏輯迴歸、SVM、隨機森林）同時進行訓練。
  - 第二層 (Meta-Model)： 將第一層所有模型的預測結果作為「新特徵」，輸入到另一個模型中進行最終訓練。
- 代表演算法：
  - 通常沒有特定的單一演算法名稱，而是一種架構設計。
  - 常見實作是第一層用多種異質模型，第二層用簡單的模型（如 Logistic Regression）來產出最終結果。
