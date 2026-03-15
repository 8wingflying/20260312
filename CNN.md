## VGG19
```python
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np

# 載入模型 (包含最後的 1000 類全連接層)
model = VGG19(weights='imagenet')

# 檢視模型結構
model.summary()
```
