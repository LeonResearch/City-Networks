import os
import torch
import argparse
import json
import torch.nn.functional as F

from copy import deepcopy
from time import time

from citynetworks import CityNetwork
from torch_geometric.datasets import Planetoid
from benchmark.configs import (
    parse_method,
    parser_add_main_args,
)
from benchmark.utils import (
    eval_acc,
    count_parameters,
    plot_logging_info
)

from torch_geometric.seed import seed_everything
from torch_geometric.loader import NeighborLoader


def train(model, loader, optimizer, device):
    model.train()
    total_loss = 0.
    all_outputs, all_labels = [], []
    for graph in loader:
        graph = graph.to(device)
        optimizer.zero_grad()
        # Note here the first graph.batch_size nodes are the seed nodes,
        # as implemented in NeighborLoader
        output = model(graph.x, graph.edge_index)[:graph.batch_size]
        labels = graph.y[:graph.batch_size]
        loss = F.cross_entropy(output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * graph.num_nodes
        all_outputs.append(output)
        all_labels.append(labels)
    acc, f1 = eval_acc(torch.cat(all_outputs), torch.cat(all_labels))
    total_loss = total_loss / len(loader.dataset)
    return acc, total_loss


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_outputs, all_labels = [], []
    for graph in loader:
        graph = graph.to(device)
        output = model(graph.x, graph.edge_index)[:graph.batch_size]
        labels = graph.y[:graph.batch_size]
        all_labels.append(labels)
        all_outputs.append(output)
    acc, f1 = eval_acc(torch.cat(all_outputs), torch.cat(all_labels))
    return acc


def main(args, logging_dict):
    seed_everything(args.seed)
    device = torch.device(f"cuda:{args.device}")

    print(f"Loading {args.dataset}...")
    if args.dataset in ["paris", "shanghai", "la", "london"]:
        dataset = CityNetwork(root=args.data_dir, name=args.dataset)
    elif args.dataset in ["cora", "citeseer"]:
        dataset = Planetoid(root=args.data_dir, name=args.dataset)
    data = dataset[0]

    # Load the big network with NeighborLoader
    train_loader = NeighborLoader(
        data,
        input_nodes=data.train_mask,
        num_neighbors=args.neighbors,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_sampling_worker,
        persistent_workers=True,
    )
    valid_loader = NeighborLoader(
        data,
        input_nodes=data.val_mask,
        num_neighbors=args.neighbors,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_sampling_worker,
        persistent_workers=True,
    )
    test_loader = NeighborLoader(
        data,
        input_nodes=data.test_mask,
        num_neighbors=args.neighbors,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_sampling_worker, 
        persistent_workers=True,
    )

    # Initialize model
    input_channels = data.x.shape[1]
    output_channels = data.y.max().item() + 1
    model = parse_method(
        args, 
        c=output_channels, 
        d=input_channels, 
        device=device
    )
    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )


    num_parameters = count_parameters(model)
    start_time = time()
    epoch_list, loss_list, train_score_list, valid_score_list, test_score_list = [], [], [], [], []
    best_acc_val, best_val_epoch = 0, 0

    print(
        f"Dataset: {args.dataset} | Num nodes: { data.num_nodes} | "
        f"Num edges: {data.num_edges} | Num node feats: {input_channels} | "
        f"Num classes: {output_channels} \n"
        f"Model: {model} | Num model parameters: {num_parameters}"
    )

    for e in range(args.epochs):
        # Train
        acc_train, tot_loss = train(model, train_loader, optimizer, device)
        # Evaluation
        if e == 0 or (e + 1) % args.display_step == 0:
            acc_val = evaluate(model, valid_loader, device)
            acc_test = evaluate(model, test_loader, device)
            # Update test scores at the best validation epoch
            if acc_val > best_acc_val:
                best_acc_val = acc_val
                test_acc_at_best_val = acc_test
                best_val_epoch = e + 1
                if args.save_model:
                    best_model = deepcopy(model)
                    model_folder = f"models/{args.experiment_name}/{args.dataset}_{args.method}"
                    os.makedirs(model_folder, exist_ok=True)
                    path = os.path.join(
                        model_folder, f"seed-{args.seed:02d}_epochs-{args.epochs}_nlayers-{args.num_layers}_nhops-{len(args.neighbors)}.pt"
                    )
                    torch.save(best_model.state_dict(), path)
            print(
                f"{args.dataset} "
                f"Seed: {args.seed:02d} "
                f"Epoch: {e+1:02d} "
                f"Loss: {tot_loss:.4f} "
                f"Train Acc: {acc_train * 100:.2f}% "
                f"Valid Acc: {acc_val * 100:.2f}% "
                f"Test Acc: {acc_test * 100:.2f}% "
                f"Time: {(time() - start_time):.2f}s "
            )
            start_time = time() # update timer 
            epoch_list.append(e+1) 
            loss_list.append(tot_loss)
            train_score_list.append(acc_train) 
            valid_score_list.append(acc_val) 
            test_score_list.append(acc_test)

    # Save the current run to a json dict
    logging_dict["seed"].append(args.seed)
    logging_dict["best_val_epoch"].append(best_val_epoch)
    logging_dict["test_at_best_val"].append(test_acc_at_best_val)
    logging_dict["epoch"].append(epoch_list) 
    logging_dict["loss"].append(loss_list)
    logging_dict["train_score"].append(train_score_list) 
    logging_dict["valid_score"].append(valid_score_list) 
    logging_dict["test_score"].append(test_score_list)

    result_folder = f"results/{args.experiment_name}/{args.dataset}_{args.method}"
    os.makedirs(result_folder, exist_ok=True)
    with open(f'{result_folder}/epochs-{args.epochs}_nlayers-{args.num_layers}_nhops-{len(args.neighbors)}.json', 'w') as file:
        print(f"Training logs and results saved at: {file.name}")
        json.dump(logging_dict, file)
    # Plot the training logs
    plot_logging_info(logging_dict, f"{result_folder}/epochs-{args.epochs}_nlayers-{args.num_layers}_nhops-{len(args.neighbors)}_training_logs.jpg")

    print(f"Seed {args.seed:02d}")
    print(f"Epoch of Best Val Acc: {best_val_epoch}")
    print(f"Test Acc at Best Val Epoch: {test_acc_at_best_val * 100:.2f}%")

    return logging_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="General Training Pipeline")
    parser_add_main_args(parser)
    args = parser.parse_args()

    args.neighbors = [-1] * args.num_layers

    logging_dict = {
        "seed":[],
        "best_val_epoch":[],
        "test_at_best_val":[],
        "epoch":[], 
        "loss":[], 
        "train_score":[], 
        "valid_score":[], 
        "test_score":[],
    }

    for seed in range(args.runs):
        args.seed = seed
        logging_dict = main(args, logging_dict)

    test_acc = torch.tensor(logging_dict['test_at_best_val']) * 100
    mean_test_acc, std_test_acc = test_acc.mean().item(), test_acc.std().item()
    logging_dict['mean_test_acc'] = round(mean_test_acc, 2)
    logging_dict['std_test_acc'] = round(std_test_acc, 2)

    result_folder = f"results/{args.experiment_name}/{args.dataset}_{args.method}/" + \
        f"epochs-{args.epochs}_nlayers-{args.num_layers}_nhops-{len(args.neighbors)}.json"
    with open(result_folder, 'w') as file:
        json.dump(logging_dict, file)
