import torch 
import torch.nn as nn

##### Model #####
    
class PlainPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PlainPredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.model(x)
    