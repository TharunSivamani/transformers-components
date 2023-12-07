# Sentence Tokenization with Transformers

## Overview

This repository provides a simple example and implementation for performing sentence tokenization using the Transformers architecture. Sentence tokenization involves breaking down a text into individual sentences.

# Code

This is a sample code to try getting hands-on with transformers

```python
from transformers import AutoTokenizer

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Example text to tokenize
text_to_tokenize = "This is an example sentence. Tokenization is a crucial NLP task."

# Tokenize the text into sentences
sentences = tokenizer.tokenize(text_to_tokenize, return_tensors="pt")

# Print the tokenized sentences
print(sentences)
```