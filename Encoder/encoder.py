import torch
import math
from torch import nn
import torch.nn.functional as F


def scaled_dot_product(q, k, v, mask=None):
    d_k = q.shape[-1]
    scaled = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    print(f"Scaled size : {scaled.shape}")
    if mask is not None:
        print(f"ADDING MASK ----------------- : {mask.shape}")
        scaled += mask
    attention = F.softmax(scaled, dim=-1)
    values = torch.matmul(attention, v)

    return values, attention


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_layer = nn.Linear(d_model, 3 * d_model)
        self.linear_layer = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, sequence_length, d_model = x.shape
        print(f"x shape : {x.shape}")

        qkv = self.qkv_layer(x)
        print(f"qkv shape : {qkv.shape}")
        qkv = qkv.reshape(
            batch_size, sequence_length, self.num_heads, 3 * self.head_dim
        )
        print(f"qkv shape after reshape : {qkv.shape}")
        qkv = qkv.permute(0, 2, 1, 3)
        print(f"qkv shape after permute : {qkv.shape}")
        q, k, v = qkv.chunk(3, dim=-1)
        print(f"q k v shape after chunking : {q.shape} - {k.shape} - {v.shape}")
        values, attention = scaled_dot_product(q, k, v, mask)
        print(f"Values shape after self attention : {values.shape}")
        values = values.reshape(
            batch_size, sequence_length, self.num_heads * self.head_dim
        )
        print(f"Values shape after reshape : {values.shape}")
        out = self.linear_layer(values)
        print(f"Out shape after Linear layer : {out.shape}")
        return out


class LayerNormalization(nn.Module):
    def __init__(self, parameter_shape, eps=1e-5):
        super(LayerNormalization, self).__init__()

        self.parameter_shape = parameter_shape
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(self.parameter_shape))
        self.beta = nn.Parameter(torch.zeros(self.parameter_shape))

    def forward(self, inputs):
        dims = [-(i + 1) for i in range(len(self.parameter_shape))]
        mean = inputs.mean(dim=dims, keepdim=True)
        print(f"Mean ({mean.size()})")
        var = ((inputs - mean) ** 2).mean(dim=dims, keepdim=True)
        std = (var + self.eps).sqrt()
        print(f"Standard Deviation  ({std.size()})")
        y = (inputs - mean) / std
        print(f"y: {y.size()}")
        out = self.gamma * y + self.beta
        print(f"self.gamma: {self.gamma.size()}, self.beta: {self.beta.size()}")
        print(f"out: {out.size()}")

        return out


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()

        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        print(f"x shape after Linear layer 1 : {x.shape}")
        x = self.relu(x)
        print(f"x shape after ReLU : {x.shape}")
        x = self.dropout(x)
        print(f"x shape after dropout layer : {x.shape}")
        x = self.linear2(x)
        print(f"x shape after Linear layer 2 : {x.shape}")
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, dropout):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNormalization([d_model])
        self.dropout1 = nn.Dropout(dropout)
        self.ffn = PositionWiseFeedForward(d_model, ffn_hidden, dropout)
        self.norm2 = LayerNormalization([d_model])
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        residual_x = x
        print("------------ ATTENTION 1 ------------")
        x = self.attention(x, mask=None)
        print("------------ DROPOUT 1 ------------")
        x = self.dropout1(x)
        print("------------ ADD AND LAYER NORM 1 ------------")
        x = self.norm1(x + residual_x)
        residual_x = x
        print("------------ ATTENTION 2 ------------")
        x = self.ffn(x)
        print("------------ DROPOUT 2 ------------")
        x = self.dropout2(x)
        print("------------ ADD AND LAYER NORM 2 ------------")
        x = self.norm2(x + residual_x)
        return x


class Encoder(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers):
        super().__init__()
        self.layers = nn.Sequential(
            *[
                EncoderLayer(d_model, ffn_hidden, num_heads, drop_prob)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        x = self.layers(x)
        return x


if __name__ == "__main__":
    d_model = 512
    num_heads = 8
    drop_prob = 0.1
    batch_size = 30
    max_sequence_length = 200
    ffn_hidden = 2048
    num_layers = 5

    encoder = Encoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers)
    x = torch.randn((batch_size, max_sequence_length, d_model))
    out = encoder(x)
    print(out)
