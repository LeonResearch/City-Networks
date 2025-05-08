# Author: Baskaran Sripathmanathan
# Link: https://openreview.net/profile?id=~Baskaran_Sripathmanathan1
import os
import argparse
import torch
import numpy as np
import torch.nn.functional as F

from benchmark.configs import parse_method, parser_add_main_args
from influence.influence_score import total_influence
from citynetworks import CityNetwork
from torch_geometric.datasets import Planetoid


def main_jacobian(model, data, args):
    jac_save_folder = args.influence_dir
    os.makedirs(jac_save_folder, exist_ok=True)
    #Calc Jacobian stuff
    print("Calculating Influence...")
    vectorize = not (args.method == "sgformer")
    tot_inf = total_influence(
        model, 
        data, 
        max_hops=16, 
        device=args.device, 
        vectorize=vectorize, 
        num_samples=args.num_samples_influence,
    )
    #Save down data
    numpy_path = os.path.join(
        jac_save_folder, f"{args.dataset}_{args.method}_influence.npy"
    )
    np.save(numpy_path,tot_inf)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="General Training Pipeline")
    parser_add_main_args(parser)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}")

    print(f"Loading {args.dataset}...")
    if args.dataset in ["paris", "shanghai", "la", "london"]:
        dataset = CityNetwork(root=args.data_dir, name=args.dataset)
    elif args.dataset in ["cora", "citeseer"]:
        dataset = Planetoid(root=args.data_dir, name=args.dataset)
    data = dataset[0]


    # Initialize model
    input_channels = data.x.shape[1]
    output_channels = data.y.max().item() + 1
    model = parse_method(
        args, 
        c=output_channels, 
        d=input_channels, 
        device=device
    )
    
    map_location_dict = {f"cuda:{i}":f"cuda:{args.device}" for i in range(1,128)}
    model_path = f"./models/{args.experiment_name}/{args.dataset}_{args.method}/" + \
        f"seed-00_epochs-{args.epochs}_nlayers-16_nhops-16.pt"
    model.load_state_dict(
        torch.load(
            model_path,
            weights_only=True,
            map_location=map_location_dict,
        )
    )
    model.eval()
    main_jacobian(model, data, args)