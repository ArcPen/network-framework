import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

import csv
from statistics import mean
import logging
from pathlib import Path
from datetime import datetime
from collections import OrderedDict

from config import Config

##### Data #####

class MNISTDataset(Dataset):
    def __init__(self, root):
        root = Path(root)
        self.load_data(root)
        self.root = root.as_posix()

    def load_data(self, root):
        csv_file = root / 'train.csv'
        label = []
        data_arr = []
        with open(csv_file) as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                label.append(int(row[0]))
                data_arr.append([int(i) for i in row[1:]])
                
        data_arr = np.array(data_arr)

        num_classes = len(set(label))
        eye = np.eye(num_classes)
        label = np.array([eye[i] for i in label]) # one-hot encoding
        
        self.label = label
        self.data = data_arr

    def __getitem__(self, idx):
        img = torch.tensor(self.data[idx]).float()
        label = torch.tensor(self.label[idx]).float()
        data_dict = OrderedDict(
            img=img,
            label=label,
            metadata=OrderedDict(
                idx=idx,
            )
        )
        return data_dict

    def __len__(self):
        return len(self.label)

    def __repr__(self):
        return f'MNISTDataset(root={self.root})'


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

##### Main #####
class MainTrainer:
    def __init__(self, args):

        self.args = args
        

        # Model
        max_epochs = args.max_epochs
        self.max_epochs = max_epochs
        self.current_epoch = 0

        # Logging
        experiment_root = Path(args.experiment_root)
        experiment_name = args.experiment_name
        save_root = experiment_root / experiment_name
        save_root.mkdir(parents=True, exist_ok=True)
        self.logger = self.get_logger(save_root)
        self.save_root = save_root.as_posix()
        self.train_log_interval = args.train_log_interval
        self.val_log_interval = args.val_log_interval

        # Load data
        self.train_loader, self.val_loader = self.load_data()

        # Load Model
        self.model = PlainPredictor(
            input_size=args.input_size,
            hidden_size=args.hidden_size,
            output_size=args.output_size
        )
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate)
        
    ### Logging ###
        
    def get_logger(self, save_root=None):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        if save_root is not None:
            log_file = (save_root / save_root.name).with_suffix('.log')
            fh = logging.FileHandler(log_file)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
            logger.info(f"Created log file, logging to {save_root}")

        return logger
    
    def log_config(self):
        ''' Log the configuration of the experiment'''
        output_list = []
        for attr in dir(self.args):
            if attr.startswith('_'):
                continue
            output_list.append(f'\n\t{attr:15} = {getattr(self.args, attr)},')
        s = ''.join(output_list)
        self.logger.info(f"Current config of the project: \n{s}")

    ### Data ###

    def load_data(self):
        ''' Load data from the dataset. As a dataloader
        '''
        logger = self.logger
        data_root = self.args.data_root

        dataset = MNISTDataset(data_root)
        all_length = len(dataset)
        data_length = self.args.data_length
        val_length = self.args.val_length

        if data_length > all_length:
            logger.warning(f"data_length larger than dataset size: {data_length}>{all_length}, "
                           "Using all data")
            data_length = all_length
        try:
            if val_length > data_length:
                raise ValueError(f"val_length larger than data_length: {val_length}>{data_length}")
        except ValueError as e:
            logger.exception(e)
            raise e

        train_length = data_length - val_length
        remaining_length = all_length - data_length

        if remaining_length > 0:
            train_dataset, val_dataset, _ = random_split(
                dataset, [train_length, val_length, remaining_length])
        else:
            train_dataset, val_dataset = random_split(dataset, [train_length, val_length])

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        logger.info(f"Loaded data from {data_root}, "
                    f"Total {all_length}, train: {train_length}, val: {val_length}")
        return train_loader, val_loader

    ### Training ###

    def train(self):
        epoch = self.current_epoch
        max_epochs = self.max_epochs
        while True:
            epoch = epoch + 1
            self.current_epoch = epoch
            should_terminate = epoch > max_epochs
            if should_terminate:
                # self.tb_writer.close()
                break

            self.logger.info(f"Starting epoch {epoch}...")
            self.train_epoch()
            self.valid()

    def train_epoch(self):
        logger = self.logger

        log_loss = []
        log_acc = []
        for batch_idx, data_dict in enumerate(self.train_loader):
            img, label = data_dict['img'], data_dict['label']
            output = self.model(img)
            
            loss = self.criterion(output, label)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Logging
            log_loss.append(loss.item())
            pred = output.argmax(dim=1, keepdim=True)
            label_truth = label.argmax(dim=1, keepdim=True)
            correct = pred.eq(label_truth).sum().item()
            acc = correct / len(label)
            log_acc.append(acc)

            if batch_idx > 0 and batch_idx % self.train_log_interval == 0:
                logger.info(f"Train epoch {self.current_epoch}, batch {batch_idx}, "
                            f"loss {mean(log_loss):.8f}, acc {mean(log_acc):.5f}")
                pass

        logger.info(f"Summary: loss {mean(log_loss):.8f}, acc {mean(log_acc):.5f}")


    def valid(self):
        pass

if __name__ == '__main__':
    
    args = Config()

    trainer = MainTrainer(args)
    trainer.train()



