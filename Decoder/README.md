# Transformer Decoder Architecture

This repository provides an in-depth explanation of the architecture of the decoder in a Transformers model. The transformer architecture, introduced by Vaswani et al. in the paper "Attention is All You Need," is widely used in natural language processing and other sequence-to-sequence tasks.

# Architecture

![decoder](https://i.stack.imgur.com/nV7Ee.jpg)

## Introduction

The Transformer model comprises an encoder-decoder architecture. This README focuses on the decoder, responsible for generating the output sequence based on the encoded input. The decoder consists of several layers, each containing masked self-attention, encoder-decoder attention, and feedforward neural network modules.

## Decoder Overview

The decoder architecture can be broken down into the following components:

1. **Input Embedding**: Convert output tokens into continuous vector representations.

2. **Positional Encoding**: Add positional information to the input embeddings to capture the sequence order.

3. **Masked Self-Attention Mechanism**: Attend to previous positions in the output sequence while preventing information flow from future positions.

4. **Encoder-Decoder Attention**: Attend to relevant information in the encoder's output to generate context-aware representations.

5. **Multi-Head Attention**: Similar to the encoder, use multiple attention heads to capture different aspects of the input sequence and context.

6. **Feedforward Neural Network**: Process the attended features through a fully connected feedforward network.

7. **Layer Normalization and Residual Connections**: Stabilize training using layer normalization and residual connections.

## Masked Self-Attention Mechanism

The masked self-attention mechanism allows the model to attend to previous positions in the output sequence while generating the current position. This prevents information flow from future positions during training.

## Encoder-Decoder Attention

The encoder-decoder attention mechanism enables the decoder to attend to relevant information in the encoder's output. It helps generate context-aware representations for each position in the output sequence.

## Positional Encoding

Similar to the encoder, positional encoding is added to the input embeddings in the decoder to provide information about the position of each token in the sequence.

## Multi-Head Attention

The multi-head attention mechanism in the decoder is employed to capture different aspects of the input sequence and context, enhancing the model's ability to learn complex relationships.

## Feedforward Neural Network

After attention mechanisms, the features are passed through a feedforward neural network in the decoder. This network applies non-linear transformations to the attended features.

## Layer Normalization and Residual Connections

Layer normalization is applied to the output of each sub-layer, followed by a residual connection. These techniques help stabilize training and facilitate the flow of gradients through the network.
