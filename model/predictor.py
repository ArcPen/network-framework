import torch 
import torch.nn as nn

##### Model #####
    
class MLPPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLPPredictor, self).__init__()
        if isinstance(hidden_size, int):
            hidden_size = [hidden_size]
        
        layers = []
        for i in range(len(hidden_size)):
            if i == 0:
                layers.append(nn.Linear(input_size, hidden_size[i]))
            else:
                layers.append(nn.Linear(hidden_size[i-1], hidden_size[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size[-1], output_size))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
