__author__ = "T. Almeida"
__email__ = "tomas.duarte.almeida@tecnico.ulisboa.pt"
__group__ = "Data-Driven Molecular Design"
__project__ = "ViralBindPredict: Empowering Viral Protein-Ligand Binding Sites through Deep Learning and Protein Sequence-Derived Insights"

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
