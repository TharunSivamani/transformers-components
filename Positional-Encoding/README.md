# Positional Encoding

## Overview

This repository contains an implementation of Positional Encoding, a critical component in Transformer models. Positional Encoding is used to provide information about the position of tokens in the input sequence, enabling the model to understand the sequential order of the data.

# Architecture

![pe](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQPde10zwBEM3t31TE57EbpypEdmCSCDz4hKQ&usqp=CAU)

## How It Works

The Positional Encoding block can be understood as follows:

1. **Positional Encoding Matrix:**
   - A matrix representing positional encodings is created and added to the input embeddings.

Formula :   

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

$$
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

where `pos` is the position, `i` is the dimension, and `d_model` is the dimension of the model

2. **Add Positional Encodings to Input Embeddings** : 
    - Sum the positional encodings with the input embeddings.

Formula :   

$$
\text{Input}_{\text{with PE}} = \text{Input}_{\text{embedding}} + \text{PE}
$$

<br>

# Usage

1. Import
Import the MultiheadAttention class into your code :

```python
from positional-encoding import PositionalEncoding
```

2. Parameters

```python
d_model=6
max_sequence_length=10
```

3. Create Instance
Create an instance of the PositionalEncoding class : 

```python
model = PositionalEncoding(d_model=6, max_sequence_length=10)
```

4. Forward Method
Use the forward method to apply the multihead attention mechanism to your input :   

```python
out = model.forward(x)
```