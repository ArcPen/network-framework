import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

import csv
from statistics import mean
import logging
from pathlib import Path
from datetime import datetime
from collections import OrderedDict

from config import Config
from model.predictor import PlainPredictor
from data.dataset import MNISTDataset


def create_model(args):
    ''' Create model based on the args. Change this if you want to
    use a different model.   
    '''
    model = PlainPredictor(
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        output_size=args.output_size
    )
    return model

##### Main #####
class MainTrainer:
    def __init__(self, args):

        self.args = args
        self.device = args.device
        
        # private variables
        self._best_acc = 0.

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

        # Load data
        self.train_loader, self.val_loader = self.load_data()

        # Log interval
        train_log_interval = args.train_log_interval
        if 0 < train_log_interval < 1:
            train_log_interval = int(len(self.train_loader) * train_log_interval)
        self.train_log_interval = train_log_interval
        val_log_interval = args.val_log_interval
        if 0 < val_log_interval < 1:
            val_log_interval = int(len(self.val_loader) * val_log_interval)
        self.val_log_interval = val_log_interval

        # Load Model
        model = create_model(args)
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate)
        
    ### Logging ###
        
    def get_logger(self, save_root=None, file_name=None):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        if save_root is not None:
            save_root = Path(save_root)
            save_root.mkdir(parents=True, exist_ok=True)
            if file_name is None:
                file_name = save_root.name
            log_file = (save_root / file_name).with_suffix('.log')
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

    ### Model Saving ###
        
    def save_model(self, prefix=None):
        if prefix is None or prefix == 'epoch':
            name = f"model_epoch_{self.current_epoch}.pth"
        else:
            name = f"model_{prefix}.pth"
        save_path = Path(self.save_root) / name
        torch.save(self.model.state_dict(), save_path)
        self.logger.info(f"Saved model to {save_path}")
    
    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
        self.logger.info(f"Loaded model from {model_path}")

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
        batch_size = self.args.batch_size

        if data_length > all_length:
            logger.warning(f"data_length larger than dataset size: {all_length} < {data_length}, "
                           "Using all data")
            data_length = all_length
        try:
            if val_length > data_length:
                raise ValueError(f"val_length larger than data_length: {data_length} < {val_length}")
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

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

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
            self.save_model('epoch')
            self.valid()

    def train_epoch(self):
        logger = self.logger
        self.model.train()

        log_loss = []
        log_acc = []
        for batch_idx, data_dict in enumerate(self.train_loader):
            img, label = data_dict['img'], data_dict['label']
            img, label = img.to(self.device), label.to(self.device)
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
                logger.info(f"Train batch {batch_idx}/{len(self.train_loader)}, "
                            f"loss {mean(log_loss):.8f}, acc {mean(log_acc):.5f}")

        logger.info(f"Summary Train epoch {self.current_epoch}:"
                    f" loss {mean(log_loss):.8f}, acc {mean(log_acc):.5f}")


    def valid(self):
        logger = self.logger
        self.model.eval()

        log_acc = []
        for batch_idx, data_dict in enumerate(self.val_loader):
            img, label = data_dict['img'], data_dict['label']
            img, label = img.to(self.device), label.to(self.device)
            output = self.model(img)
            
            # Logging
            pred = output.argmax(dim=1, keepdim=True)
            label_truth = label.argmax(dim=1, keepdim=True)
            correct = pred.eq(label_truth).sum().item()
            acc = correct / len(label)
            log_acc.append(acc)

            do_log = (batch_idx > 0 and batch_idx % self.val_log_interval == 0)

            if do_log:
                logger.info(f"Val batch {batch_idx}/{len(self.val_loader)}, "
                            f"acc {mean(log_acc):.5f}")
                pass
        
        curr_acc = mean(log_acc)
        logger.info(f"Summary Val epoch {self.current_epoch}: acc {curr_acc:.5f}")

        if curr_acc > self._best_acc:
            self._best_acc = curr_acc
            self.save_model('best')
            logger.info(f"New best model saved with acc {curr_acc:.5f}")


class MainTester(MainTrainer):
    ''' The test and training process should be two separated processes, where in one you
    have GT and in the other you don't. So they are seperated into two classes.

    Possible methods of usage:
    1. Pass the args from an training process to it, note that config would change
       each time generated.
    2. Use same args as training process, but provide model_path to specify model 
       to load. In this case, folder of the model is used as save_root.

    '''
    def __init__(self, args, model_path=None):
        self.args = args
        self.device = args.device

        if model_path is None:
            save_root = Path(args.experiment_root) / args.experiment_name
            model_path = save_root / 'model_best.pth'
        else:
            model_path = Path(model_path)
            save_root = model_path.parent
        save_root = save_root / 'test_out'
        self.logger = self.get_logger(
            save_root, file_name=args.experiment_name)
        self.save_root = save_root.as_posix()
        
        # Model
        model = create_model(args)
        self.model = model.to(self.device)
        self.load_model(model_path)
        
        test_loader = self.load_data()
        self.test_loader = test_loader

        test_log_interval = args.test_log_interval
        if 0 < test_log_interval < 1:
            test_log_interval = int(len(test_loader) * test_log_interval)
        self.test_log_interval = test_log_interval

    def load_data(self):
        data_root = self.args.data_root
        batch_size = self.args.batch_size

        dataset = MNISTDataset(data_root, mode='test')
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return loader
        
    def test(self):
        logger = self.logger
        self.model.eval()

        results = []
        for batch_idx, data_dict in enumerate(self.test_loader):
            img, idx = data_dict['img'], data_dict['metadata']['idx']
            img = img.to(self.device)
            output = self.model(img)
            
            # Logging
            pred = output.argmax(dim=1, keepdim=False)
            results.extend(zip(idx.tolist(), pred.tolist()))

            do_log = (batch_idx > 0 and batch_idx % self.test_log_interval == 0)
            if do_log:
                logger.info(f"Test, batch {batch_idx}/{len(self.test_loader)}")
        
        logger.info(f"Saving results...")

        save_path = Path(self.save_root) / self.args.experiment_name
        save_path = save_path.with_suffix('.csv')
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['ImageId', 'Label'])
            writer.writerows(results)
        logger.info(f"Results saved to {save_path}")


    def train(self):
        raise NotImplementedError("Test process don't have this.")
    
    def valid(self):
        raise NotImplementedError("Test process don't have this.")


if __name__ == '__main__':
    
    args = Config()

    trainer = MainTrainer(args)
    trainer.train()

    model_path = '/disk1/user3/workspace/kaggle/0525-mnist-digits/experiments/project-testing/exp_0529_010011/model_best.pth'
    tester = MainTester(args, model_path=model_path)
    tester.test()



