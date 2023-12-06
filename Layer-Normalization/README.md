# Layer - Normalization

This repository provides an implementation of Layer Normalization within a Transformers block using PyTorch. Layer Normalization is a technique that helps stabilize the training of deep neural networks by normalizing the inputs to a layer.

# Architecture

![layer-norm](https://cdn.sanity.io/images/vr8gru94/production/68cddd98ed9529e2b0edac143a47ec1b5ecbadd3-800x521.png)

# Math 

![layer-norm](https://melfm.github.io/posts/2018-08-Understanding-Normalization/img/batch_norm_forward.png)

## Internal Steps

### 1. Calculate Mean, Variance, and Standard Deviation

Given a tensor x of shape batch_size, hidden_dim , calculate the mean `mu` , variance `var`, and standard deviation `std` across the feature dimension:

$$
\mu = \frac{1}{\text{{hidden-size}}} \sum_{i=1}^{\text{{hidden-size}}} x_i
$$

<br> 

$$
\sigma^2 = \frac{1}{\text{{hidden-size}}} \sum_{i=1}^{\text{{hidden-size}}} (x_i - \mu)^2
$$

<br> 

$$
\sigma = \sqrt{\sigma^2 + \epsilon}
$$

### 2. Normalize the Features

Normalize the input tensor x using the computed mean `mu`, standard deviation `std`, learnable scale `gamma`, and learnable bias `beta`:

$$
\text{{Layer Normalization}}(x) = \gamma \cdot \frac{x - \mu}{\sigma} + \beta
$$

where `gamma` and `beta` are learnable parameters.

## Implementation

The Layer Normalization is implemented in the `layer-normalization.py` file. The implementation follows the standard formulation of Layer Normalization.

```python
from layer-normalization import LayerNormalization

batch_size = 3
sentence_length = 4 # My name is XXXX
embedding_dim = 8

inputs = torch.randn(embedding_dim, batch_size, sentence_length)

layer_norm = LayerNormalization(inputs.shape[1:])
out = layer_norm(inputs)

print(out)
print(out.shape)
```
