import os
import json
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Optional 
from copy import deepcopy


@dataclass
class TrainLogger:
    epochs:      List[int]   = field(default_factory=list)
    train_loss:  List[float] = field(default_factory=list)
    train_acc:   List[float] = field(default_factory=list)
    val_loss:    List[float] = field(default_factory=list)
    val_acc:     List[float] = field(default_factory=list)
    test_loss:   List[float] = field(default_factory=list)
    test_acc:    List[float] = field(default_factory=list)
    best_val_acc: float = field(default=0.0, init=False)
    best_epoch: Optional[int] = field(default=0, init=False)
    test_acc_at_best_val: Optional[float] = field(default=0.0, init=False)
    
    # Main logging function
    def log_epoch(
        self,
        configs: dict,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        test_loss: Optional[float] = None,
        test_acc: Optional[float] = None,
        time_elapsed: Optional[float] = None,
        model: torch.nn.Module = None,
        plot_logs: bool = False,
    ):
        self.epochs.append(epoch)
        self.train_loss.append(train_loss)
        self.train_acc.append(train_acc)
        self.val_loss.append(val_loss)
        self.val_acc.append(val_acc)
        self.test_loss.append(float("nan") if test_loss is None else test_loss)
        self.test_acc.append(float("nan") if test_acc  is None else test_acc)
        
        # Update Test Acc at Best Val epoch
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_epoch = epoch
            self.test_acc_at_best_val = test_acc
            # Save the model checkpoint at this epoch
            if configs['train']['save_model']:
                best_model = deepcopy(model)
                model_folder = f"{configs['models_dir']}/{configs['exp_name']}/" + \
                    f"{configs['dataset']}_{configs['method']}"
                os.makedirs(model_folder, exist_ok=True)
                path = os.path.join(
                    model_folder,
                    f"seed-{configs['seed']:02d}_epochs-{configs['train']['epochs']}_" + \
                        f"nlayers-{configs['model']['num_layers']}.pt"
                )
                torch.save(best_model.state_dict(), path)
    
        print(
            f"{configs['dataset']} "
            f"Seed: {configs['seed']:02d} "
            f"Epoch: {epoch:02d} "
            f"Train Loss: {train_loss:.4f} "
            f"Train Acc: {train_acc:.2f}% "
            f"Valid Acc: {val_acc:.2f}% "
            f"Test Acc: {test_acc:.2f}% "
            f"Time: {(time_elapsed):.2f}s "
        )
        
    def to_dict(self):
        return asdict(self)

    def save_to_json(self, path: str):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)


def load_config(path):
    path = Path(path).expanduser().resolve()
    with path.open("r") as fh:
        return yaml.safe_load(fh)


def get_result_path(configs):
    result_folder = f"{configs['results_dir']}/{configs['exp_name']}/" + \
        f"{configs['dataset']}_{configs['method']}"
    file_name = f"epochs-{configs['train']['epochs']}_" + \
        f"nlayers-{configs['model']['num_layers']}"
    return result_folder, file_name


def load_results(configs):    
    result_folder, file_name = get_result_path(configs)
    with open(f'{result_folder}/{file_name}.json', 'r') as file:
        logging_results = json.load(file)
    return logging_results


def save_results(configs, logging_results, save_configs=False):
    result_folder, file_name = get_result_path(configs)
    os.makedirs(result_folder, exist_ok=True)
    with open(f'{result_folder}/{file_name}.json', 'w') as file:
        json.dump(logging_results, file)
    if save_configs:
        with open(f'{result_folder}/{file_name}_configs.yaml', "w") as f:
            yaml.dump(configs, f)


def plot_logging_info(logging_results, configs):
    epochs = np.array(logging_results['0']['epochs'])  # all epoch lists are the same
    # Calculating mean and std
    def mean_and_std(data_key):
        values = [logging_results[key][data_key] for key in logging_results.keys()]
        values = np.array(values)
        mean = np.mean(values, axis=0)
        std = np.std(values, axis=0)
        return mean, std

    colors = {
        'train_loss': 'blue',
        'train_acc': 'blue',
        'val_loss': 'red',
        'val_acc': 'red',
        'test_loss': 'green',
        'test_acc': 'green',
    }

    fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharey=False)

    # Subplot 1 (Left): Loss
    for metric in ['train_loss', 'val_loss', 'test_loss']:
        mean, std = mean_and_std(metric)
        axs[0].plot(epochs, mean, label=f'{metric}', color=colors[metric], linewidth=2)
        axs[0].fill_between(epochs, mean - std, mean + std, color=colors[metric], alpha=0.2)
    
    axs[0].set_xlabel('Epoch', fontsize=12)
    axs[0].set_ylabel('Loss', fontsize=12)
    axs[0].set_title('Train / Valid / Test Loss over Epochs', fontsize=14, fontweight='bold')
    axs[0].grid(True, linestyle='--', alpha=0.7)

    # Subplot 2 (Right): Acc
    for metric in ['train_acc', 'val_acc', 'test_acc']:
        mean, std = mean_and_std(metric)
        axs[1].plot(epochs, mean, label=f'{metric}', color=colors[metric], linewidth=2)
        axs[1].fill_between(epochs, mean - std, mean + std, color=colors[metric], alpha=0.2)

    axs[1].set_xlabel('Epoch', fontsize=12)
    axs[1].set_ylabel('Acc', fontsize=12)
    axs[1].set_title('Train / Valid / Test Acc over Epochs', fontsize=14, fontweight='bold')
    axs[1].legend(loc='best')
    axs[1].grid(True, linestyle='--', alpha=0.7)

    result_folder, file_name = get_result_path(configs)
    plot_path = f"{result_folder}/{file_name}_training_logs.jpg"

    plt.tight_layout()
    plt.savefig(plot_path)