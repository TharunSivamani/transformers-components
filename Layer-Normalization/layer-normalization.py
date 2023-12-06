import torch
import torch.nn as nn

class LayerNormalization(nn.Module):

    def __init__(self, parameter_shape, eps = 1e-5):
        super(LayerNormalization, self).__init__()

        self.parameter_shape = parameter_shape
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(parameter_shape))
        self.beta = nn.Parameter(torch.zeros(parameter_shape))

    def forward(self, inputs):

        dims = [ -(i + 1) for i in range(len(self.parameter_shape))]

        mean = inputs.mean(dim = dims, keepdim = True)
        var = ((inputs - mean) ** 2).mean(dim = dims, keepdim = True)
        std = (var + self.eps).sqrt()

        y = (inputs - mean) / std

        out = self.gamma * y + self.beta
        return out
    
if __name__ == "__main__":

    batch_size = 3
    sentence_length = 4 # My name is XXXX
    embedding_dim = 8

    inputs = torch.randn(embedding_dim, batch_size, sentence_length)

    layer_norm = LayerNormalization(inputs.shape[1:])
    out = layer_norm(inputs)

    print(out)
    print(out.shape)