from torch import nn
from torch import optim




ACTIVATIONS = {
    'relu': nn.ReLU,
    'leaky-relu': nn.LeakyReLU,
    'gelu': nn.GELU,
    'sigmoid': nn.Sigmoid,
    'tanh': nn.Tanh
}




CRITERIONS = {
    'mse': nn.MSELoss,
    'bce-logits': nn.BCEWithLogitsLoss
}




OPTIMIZERS = {
    'adam': optim.Adam,
    'sgd': optim.SGD
}
