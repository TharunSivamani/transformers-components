# Multi-Head-Attention

This repository contains a PyTorch implementation of Multihead Attention, a key component in transformer models. Multihead Attention allows the model to attend to different parts of the input sequence independently, enhancing its ability to capture complex patterns in the data.

# Architecture

![multi-head-attn](https://production-media.paperswithcode.com/methods/Screen_Shot_2020-07-08_at_12.17.05_AM_st5S0XV.png)

# Usage

1. Import
Import the MultiheadAttention class into your code :

```python
from multi-head-attention import MultiHeadAttention
```

2. Parameters

```python
input_dim = 1024
d_model = 512
num_heads = 8
batch_size = 30
sequence_length = 5
```

3. Create Instance
Create an instance of the MultiheadAttention class : 

```python
model = MultiHeadAttention(input_dim, d_model, num_heads)
```

4. Forward Method
Use the forward method to apply the multihead attention mechanism to your input :   

```python
out = model.forward(x)
```