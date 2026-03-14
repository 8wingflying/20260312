# DistilBert
```python
from transformers import DistilBertTokenizer

# 加載 DistilBERT 預訓練的分詞器
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# 對文本進行分詞
text = "I love natural language processing!"
tokens = tokenizer(text, return_tensors="pt")

print(tokens)
```

# TinyBERT
```python
from transformers import AutoTokenizer, AutoModel
import torch

# 調用華為提供的 4 層 TinyBERT (通用版本)
model_id = "huawei-noah/TinyBERT_General_4L_312D"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id)

text = "Hugging Face is a great platform for NLP."
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

# 獲取最後一層的隱藏狀態
last_hidden_states = outputs.last_hidden_state
print(f"向量維度: {last_hidden_states.shape}") 
# 輸出通常為 [1, sequence_length, 312]，BERT-Base 則是 768
```
