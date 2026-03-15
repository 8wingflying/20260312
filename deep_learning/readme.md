# deep_learning <===神經網路
- CNN ==> 電腦視覺(Computer vision)
- RNN ==> NLP自然語言處理|Natural Language Processing
- GenAI
- `深度`強化學習(`Deep` Q-learning) <==強化學習(Q-learning)

## 神經網路 1: `感知機` perceptron模型 ==> `多層`感知機 `Multilayer` perceptron, MLP
- NN(neural network)
- `感知機` perceptron模型
  - 激活函數 Activation function f(z)
  - 模型參數(不是`超參數`):W(權重)  b(bias偏差值) <==演算法訓練的
  - XOR PROBLEM ==>`多層`感知機 `Multilayer` perceptron, MLP ==> 有中間層(Hidden Layer)
- `機器` `學習`
  - 誤差函數(Error function) ==> 誤差函數`最小化` ==> 演算法==Gradient Descent`梯度`下降(負號)法
    - 模型`參數`(Parameters):Weight(權重) Bias(偏差值) <==演算法訓練的
    - 模型`超參數`(Hyperparameter): η = 學習率(learning rate) 正數, 約取0.1~0.3 事先要給定的
  - 每一輪的[w,b]參數更新 ==>Gradient Descent`梯度`下降(負號)法
    - (誤差)backpropagation`反向`傳播
    - Forward propagation(正向傳播)
 - MLP應用
   - https://github.com/8wingflying/20260312/blob/main/code/MLP.md
   - 分類(Classification)問題 ==> iris (三元分類)
   - 迴歸(regression)問題 ==> Boston| 加州房價預測問題
 
 ## 神經網路 2: Deep Learning
 


