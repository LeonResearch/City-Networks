import torch
import argparse
import torch.nn.functional as F
from time import time
from torch_geometric.seed import seed_everything
from torch_geometric.utils import dropout_edge, degree
from torch_geometric.transforms import GDC

from benchmark.GT.positional_encoding import positional_encoding
from benchmark.initialization import (
    parse_method,
    parser_add_main_args,
)
from utils import (
    load_dataset,
    eval_acc,
    count_parameters,
    GDC_transform,
)
from loggers import (
    TrainLogger,
    load_config,
    load_results,
    save_results,
    plot_logging_info,
)

def train(configs, model, data, train_mask, optimizer, device):
    model.train()
    data = data.to(device)
    labels = data.y
    optimizer.zero_grad()
    if configs['method'] == 'dropedge':
        edge_index, _ = dropout_edge(data.edge_index, p=0.2, training=True)
    else:
        edge_index = data.edge_index
    output = model(data.x, edge_index)
    if configs['method'] == 'nodeformer':
        output, loss_add = output
    else:
        loss_add = [0.]
    output = output.log_softmax(dim=-1)
    loss = F.nll_loss(output[train_mask], labels[train_mask])
    loss += loss_add[0]

    loss.backward()
    optimizer.step()
    acc, f1 = eval_acc(output[train_mask], labels[train_mask])
    return acc, loss.item()


@torch.no_grad()
def evaluate(model, data, mask, device):
    model.eval()
    data = data.to(device)
    labels = data.y
    output = model(data.x, data.edge_index)
    
    if configs['method'] == 'nodeformer':
        output, loss_add = output
    else:
        loss_add = [0.]
    
    output = output.log_softmax(dim=-1)
    
    loss = F.nll_loss(output[mask], labels[mask])
    loss += loss_add[0]

    acc, f1 = eval_acc(output[mask], labels[mask])
    return acc, loss.item()

def main(configs):
    device = torch.device(f"cuda:{configs['device']}")
    seed = configs['seed']
    data = load_dataset(configs)
    # Data pre-transformation for GDC method
    if configs['method'] == 'gdc':
        data = GDC_transform(data)
    elif configs['method'] == 'gps':
        data = positional_encoding(data, configs['model']['pe_type'])

    # Degree distribution for PNAConv method
    deg = degree(data.edge_index[1], data.num_nodes, dtype=torch.long)
    deg_hist = torch.bincount(deg, minlength=int(deg.max().item()) + 1).float()
    
    # Initialize model
    input_channels = data.x.shape[1]
    output_channels = data.y.max().item() + 1
    model = parse_method(
        configs, 
        c=output_channels, 
        d=input_channels,
        deg=deg_hist,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=configs['train']['lr'],
        weight_decay=configs['train']['weight_decay'],
    )

    num_parameters = count_parameters(model)
    print(
        f"Dataset: {configs['dataset']} | Num nodes: { data.num_nodes} | "
        f"Num edges: {data.num_edges} | Num node feats: {input_channels} | "
        f"Num classes: {output_channels} \n"
        f"Model: {model} | Num model parameters: {num_parameters}\n"
    )

    logger = TrainLogger()
    start_time = time()

    for e in range(1, configs['train']['epochs'] + 1):
        # Train
        acc_train, loss_train = train(configs, model, data, data.train_mask, optimizer, device)
        # Evaluation
        if e == 1 or e % configs['train']['evaluation_window'] == 0:
            acc_val, loss_val = evaluate(model, data, data.val_mask, device)
            acc_test, loss_test = evaluate(model, data, data.test_mask, device)
            time_elapsed = time() - start_time
            start_time = time()
            logger.log_epoch(
                configs = configs,
                epoch = e,
                train_loss = loss_train,
                train_acc = acc_train,
                val_loss = loss_val,
                val_acc = acc_val,
                test_loss = loss_test,
                test_acc = acc_test,
                time_elapsed = time_elapsed,
                model = model, 
                plot_logs = True,
            )
            # Load and then save the current results to a json dict
            logging_results = load_results(configs)
            logging_dict = logger.to_dict()
            logging_results[str(seed)] = logging_dict # json key is always str
            save_results(configs, logging_results)

    # Print final results and plot the training logs
    print(
        f"Seed {seed:02d} | Best Val Epoch: {logging_dict['best_epoch']} | "
        f"Val Acc at Epoch {logging_dict['best_epoch']}: {logging_dict['best_val_acc']:.2f}% | "
        f"Test Acc at Epoch {logging_dict['best_epoch']}: {logging_dict['test_acc_at_best_val']:.2f}% | "
    )
    plot_logging_info(logging_results, configs)

    return logging_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="General Training Pipeline")
    parser_add_main_args(parser)
    args = parser.parse_args()

    config_path = f"./configs/{args.dataset}/{args.method}.yaml"
    configs = load_config(config_path)
    configs["exp_name"] = args.exp_name
    configs['dataset'] = args.dataset
    configs['method'] = args.method
    configs['model']['num_layers'] = args.num_layers
    if args.hidden_size > 0:
        configs['model']['hidden_channels'] = args.hidden_size
    configs['device'] = args.device

    logging_results = {}
    save_results(configs, logging_results, save_configs=True)

    for seed in range(args.runs):
        configs['seed'] = seed
        seed_everything(seed)
        logging_results = main(configs)

    test_acc = [logging_results[seed]['test_acc_at_best_val'] for seed in logging_results.keys()]
    test_acc = torch.tensor(test_acc)
    mean_test_acc, std_test_acc = test_acc.mean().item(), test_acc.std().item()
    logging_results['mean_test_acc'] = round(mean_test_acc, 1)
    logging_results['std_test_acc'] = round(std_test_acc, 2)
    save_results(configs, logging_results)
