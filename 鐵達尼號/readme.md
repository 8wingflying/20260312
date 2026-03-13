## 鐵達尼號資料集（Titanic Dataset） 
- 鐵達尼號資料集（Titanic Dataset）是機器學習領域中最經典的入門資料集之一，通常託管於 Kaggle 平台。
- 它的核心任務是根據乘客的個人資訊（如年齡、性別、艙等），預測該乘客是否在 1912 年的船難中生存。
- 這個資料集非常適合練習資料清洗（Data Cleaning）、**特徵工程（Feature Engineering）以及分類模型（Classification）**的實作。

## 📊 資料欄位說明（Data Dictionary）
- 資料集通常分為 train.csv（訓練集，含解答）與 test.csv（測試集，不含解答）。主要欄位如下：
- 欄位名稱	說明	取值範例
  - PassengerId	乘客編號	1, 2, 3...
  - [答案]Survived	目標變數：是否生還	0 = 否, 1 = 是
  - Pclass	客艙等級（社會經濟地位）	1 = 頭等, 2 = 二等, 3 = 三等
  - Name	乘客姓名	Braund, Mr. Owen Harris
  - Sex	性別	male, female
  - Age	年齡	22, 38, 0.42 (嬰兒)
  - SibSp	船上的兄弟姐妹/配偶人數	0, 1, 2...
  - Parch	船上的父母/子女人數	0, 1, 2...
  - Ticket	船票號碼	A/5 21171, PC 17599
  - Fare	票價	7.25, 71.2833
  - Cabin	船艙號碼	C85, C123
  - Embarked	登船港口	C=瑟堡, Q=昆士敦, S=南安普敦

## 🔍 資料集的核心挑戰
- 處理此資料集時，你通常會遇到以下幾個關鍵問題：
- 缺失值處理（Handling Missing Values）：
  - Age 欄位約有 20% 缺失，通常用中位數或根據稱謂（Mr./Mrs.）填補。
  - Cabin 缺失率極高（約 77%），實務上常選擇刪除此欄位或僅提取第一個字母（代表樓層）。
  - Embarked 有極少數缺失，通常用眾數（最多人登船的港口）填補。

## 特徵工程（Feature Engineering）：
- 稱謂提取： 從 Name 中提取 Mr., Miss, Master 等資訊，這對預測生存率非常有幫助。
- 家庭規模： 將 SibSp 與 Parch 相加，判斷乘客是獨自一人還是團體行動。

## 關鍵觀察：
- 女性與小孩： 由於「婦孺優先」政策，女性的生存率遠高於男性。
- 艙等： 頭等艙（Pclass 1）乘客的生存機會顯著高於三等艙。
