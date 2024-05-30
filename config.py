from typing import Union
from datetime import datetime
from pathlib import Path

class Config:
    project_name:str = 'project-testing'
    experiment_prefix:str = 'exp'
    
    device:str = 'cuda'

    # Data
    data_root:str = 'dataset'

    # Model
    max_epochs:int = 10
    batch_size:int = 32
    learning_rate:float = 0.001
    input_size:int = 28*28
    hidden_size:int = 128
    output_size:int = 10

    # Logging
    train_log_interval:int = 200
    val_log_interval:int = 50
    test_log_interval: Union[int, float] = 0.1 # int for fixed interval, float for percentage
    data_length:int = 42000
    val_length:int = 1000

    # Autimaticallg set attributes
    experiment_root:str = None
    experiment_name:str = None

    def __init__(self, **kwargs):
        self._set_experiment_name()
        for k, v in kwargs.items():
            setattr(self, k, v)

    def _set_experiment_name(self):
        experiment_root = Path(".") / 'experiments' / self.project_name
        timestamp = datetime.now().strftime('%m%d_%H%M%S')
        experiment_name = f"{self.experiment_prefix}_{timestamp}"

        self.experiment_root = experiment_root.as_posix()
        self.experiment_name = experiment_name


 