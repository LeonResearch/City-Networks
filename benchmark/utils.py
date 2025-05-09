import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score


def eval_acc(output, labels):
    # Ensure preds is a 1D array of predicted label indices
    preds = output.argmax(dim=-1).cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    # Compute macro F1 score and accuracy
    macro_f1 = f1_score(labels, preds, average='macro')
    acc = (preds == labels).sum() / preds.shape[0]
    return acc, macro_f1


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_logging_info(logging_dict, path):
    epochs = np.array(logging_dict['epoch'][0])  # all epoch lists are the same
    # Calculating mean and std
    def mean_and_std(data_key):
        values = np.array(logging_dict[data_key])
        mean = np.mean(values, axis=0)
        std = np.std(values, axis=0)
        return mean, std

    colors = {
        'loss': 'red',
        'train_score': 'blue',
        'valid_score': 'brown',
        'test_score': 'green'
    }

    fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharey=False)
    mean_loss, std_loss = mean_and_std('loss')

    # Subplot 1 (Left): Loss
    axs[0].plot(epochs, mean_loss, label='loss', color='red', linewidth=2)
    axs[0].fill_between(epochs, mean_loss - std_loss, mean_loss + std_loss, color='red', alpha=0.2)
    axs[0].set_xlabel('Epoch', fontsize=12)
    axs[0].set_ylabel('Loss', fontsize=12)
    axs[0].set_title('Loss over Epochs', fontsize=14, fontweight='bold')
    axs[0].grid(True, linestyle='--', alpha=0.7)

    # Subplot 2 (Right): Other metrics
    for metric in ['train_score', 'valid_score', 'test_score']:
        mean, std = mean_and_std(metric)
        axs[1].plot(epochs, mean, label=f'{metric}', color=colors[metric], linewidth=2)
        axs[1].fill_between(epochs, mean - std, mean + std, color=colors[metric], alpha=0.2)

    axs[1].set_xlabel('Epoch', fontsize=12)
    axs[1].set_ylabel('Score', fontsize=12)
    axs[1].set_title('Training, Validation, and Test Scores over Epochs', fontsize=14, fontweight='bold')
    axs[1].legend(loc='best')
    axs[1].grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(path)