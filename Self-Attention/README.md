# Transformers Self-Attention Block

## Overview

This repository contains a practical implementation of the self-attention block commonly used in Transformer models. Self-attention is a key mechanism in Transformers that allows the model to weigh different parts of the input sequence differently when making predictions.

# Terminologies

1. Q : What I am looking for    
    [ sequence_length x d_q ]

2. K : What I can offer     
    [ sequence_length x d_k ]

3. V : What I actually offer     
    [ sequence_length x d_v ]

# Architecture

![sself-attn-img](https://vaclavkosar.com/images/expire-span-attention-recap.png)

## How It Works

The self-attention block consists of three main components:

1. **Query, Key, and Value Matrices:**
   - Input sequence is represented as three matrices: Q (Query), K (Key), and V (Value).
   - These matrices are obtained by linear transformations of the input sequence.

   Formulas:
   
    $$
    Q = X \cdot W_q
    $$
   
    $$
    K = X \cdot W_k
    $$
   
    $$
    V = X \cdot W_v   
    $$


where `X` is the input sequence, and `W_q`, `W_k`, and `W_v` are learnable weight matrices.

2. **Attention Scores:**
- Calculate the attention scores by taking the dot product of the Query and Key matrices and scaling by the square root of the dimension of the key vectors.

    Formulas:
  
$$   
\text{Attention Scores} = \text{softmax}\left(\frac{Q \cdot K^T}{\sqrt{d_k}}\right)    
$$   



where `d_k` is the dimension of the key vectors.

3. **Weighted Sum:**
- Multiply the attention scores by the Value matrix to get the weighted sum, which is the output of the self-attention block.

    Formula:

$$  
\text{Output} = \text{Attention Scores} \cdot V
$$  


  

