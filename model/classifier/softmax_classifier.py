from torch import nn, optim
from ..base_model import base_model

class Softmax_Layer(base_model):
    """
    Softmax classifier for sentence-level relation extraction.
    """

    def __init__(self, input_size, num_class):
        """
        Args:
            num_class: number of classes
        """
        super(Softmax_Layer, self).__init__()
        self.input_size = input_size
        self.num_class = num_class
        self.fc = nn.Linear(self.input_size, self.num_class, bias=False)

    def forward(self, input):
        """
        Args:
            args: depends on the encoder
        Return:
            logits, (B, N)
        """
        logits = self.fc(input)
        return logits
