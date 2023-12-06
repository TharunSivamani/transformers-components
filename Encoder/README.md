# Transformer Encoder Architecture

This repository provides an in-depth explanation of the architecture of the encoder in a Transformers model. Transformers, introduced by Vaswani et al. in the paper "Attention is All You Need," have become a cornerstone in natural language processing and other sequence-to-sequence tasks.

# Architecture

![encoder](https://builtin.com/sites/www.builtin.com/files/styles/ckeditor_optimize/public/inline-images/Transformer-neural-network-14.png)

## Introduction

The Transformer model consists of an encoder-decoder architecture. This README focuses on the encoder part, which is responsible for processing the input sequence. The encoder is composed of multiple layers, each containing a self-attention mechanism and feedforward neural network.

## Encoder Overview

The encoder architecture can be broken down into the following components:

1. **Input Embedding**: Convert input tokens into continuous vector representations.

2. **Positional Encoding**: Add positional information to the input embeddings to capture the sequence order.

3. **Multi-Head Self-Attention Mechanism**: Attend to different parts of the input sequence simultaneously.

4. **Feedforward Neural Network**: Process the attended features through a fully connected feedforward network.

5. **Layer Normalization and Residual Connections**: Stabilize training using layer normalization and residual connections.

## Self-Attention Mechanism

The self-attention mechanism allows the model to weigh different words in the input sequence differently. It computes attention scores between all pairs of words in the sequence, producing a weighted sum for each word. This mechanism enables the model to focus on relevant context words for each position.

## Positional Encoding

As transformers lack inherent sequential information, positional encoding is added to the input embeddings to provide information about the position of each token in the sequence. This allows the model to consider the order of tokens.

## Multi-Head Attention

Multi-Head Attention extends the self-attention mechanism by employing multiple attention heads. Each head learns different aspects of the input sequence, enabling the model to capture a richer set of relationships.

## Feedforward Neural Network

After attention, the features are passed through a feedforward neural network. This network consists of fully connected layers and applies non-linear transformations to the attended features.

## Layer Normalization and Residual Connections

Layer normalization is applied to the output of each sub-layer, followed by a residual connection. These help in stabilizing training and facilitate the flow of gradients through the network.