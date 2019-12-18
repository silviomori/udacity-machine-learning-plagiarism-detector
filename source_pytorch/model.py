import torch
import torch.nn as nn
import torch.nn.functional as F


## DONE: Complete this classifier
class BinaryClassifier(nn.Module):
    """
    Define a neural network that performs binary classification.
    The network should accept your number of features as input, and produce 
    a single sigmoid value, that can be rounded to a label: 0 or 1, as output.

    Notes on training:
    To train a binary classifier in PyTorch, use BCELoss.
    BCELoss is binary cross entropy loss, documentation:
    https://pytorch.org/docs/stable/nn.html#torch.nn.BCELoss
    """

    ## DONE: Define the init function, the input params are required
    # (for loading code in train.py to work)
    def __init__(self, input_features, hidden_dim, output_dim, dropout=0.2):
        """
        Initialize the model by setting up linear layers.
        Use the input parameters to help define the layers of your model.

        Arguments:
        :param input_features: the number of input features in your \
            training/test data
        :param hidden_dim: define the number of nodes in the hidden layer(s)
        :param output_dim: the number of outputs you want to produce
        :param dropout: probability of an element to be zeroed. Default: 0.2
        """
        super(BinaryClassifier, self).__init__()

        nodes = []
        nodes.append(input_features)
        if type(hidden_dim) == int:
            nodes.append(hidden_dim)
        elif type(hidden_dim) == list:
            nodes.extend(hidden_dim)
        nodes.append(output_dim)

        self.module_list = nn.ModuleList()
        for n_in, n_out in zip(nodes[:-1], nodes[1:]):
            self.module_list.append(nn.Linear(n_in, n_out))

        self.dropout = nn.Dropout(dropout)

    ## DONE: Define the feedforward behavior of the network
    def forward(self, x):
        """
        Perform a forward pass of our model on input features, x.

        Arguments:
        :param x: A batch of input features of size (batch_size,input_features)

        Return:
        :return: A single, sigmoid-activated value as output
        """

        # define the feedforward behavior
        for layer in self.module_list[:-1]:
            x = F.relu(layer(x))
            x = self.dropout(x)
        x = self.module_list[-1](x)

        return torch.sigmoid(x)
