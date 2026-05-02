import os
import argparse
import torch
import numpy as np

from influence import total_influence
from benchmark.GT.positional_encoding import positional_encoding
from benchmark.initialization import parse_method, parser_add_main_args
from utils import load_dataset
from loggers import load_config
from time import time

def main_jacobian(model, data, device, args):
    os.makedirs(args.influence_dir, exist_ok=True)
    #Calc Jacobian stuff
    print("Calculating Influence...")
    vectorize = not (args.method in ["sgformer", "exphormer"])
    avg_tot_inf, R = total_influence(
        model, 
        data, 
        max_hops=args.num_layers, 
        num_samples=args.num_samples_influence,
        normalize=True,
        average=True,
        device=device, 
        vectorize=vectorize, 
    )
    
    #Save to result_dir 
    numpy_path = os.path.join(
        args.influence_dir, f"{args.dataset}_{args.method}_avg_tot_inf.npy"
    )
    np.save(numpy_path, avg_tot_inf)
    numpy_path = os.path.join(
        args.influence_dir, f"{args.dataset}_{args.method}_R.npy"
    )
    np.save(numpy_path, R)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="General Training Pipeline")
    parser_add_main_args(parser)
    args = parser.parse_args()
    device_name = "cpu" if args.device < 0 else f"cuda:{args.device}"
    device = torch.device(device_name)
    map_location_dict = {f"cuda:{i}":device_name for i in range(1,128)}

    config_path = f"./configs/{args.dataset}/{args.method}.yaml"
    configs = load_config(config_path)
    configs["exp_name"] = args.exp_name
    configs['dataset'] = args.dataset
    configs['method'] = args.method
    configs['model']['num_layers'] = args.num_layers
    configs['model']['hidden_channels'] = args.hidden_size
    configs['device'] = args.device
    configs['seed'] = 0

    data = load_dataset(configs)
    if configs['method'] == 'gps':
        data = positional_encoding(data, configs['model']['pe_type'])

    # Initialize model
    input_channels = data.x.shape[1]
    output_channels = data.y.max().item() + 1
    model = parse_method(
        configs, 
        c=output_channels, 
        d=input_channels,
        device=device,
    )

    model_path = f"./models/{configs['exp_name']}/{configs['dataset']}_{configs['method']}/" + \
        f"seed-00_epochs-{configs['train']['epochs']}_nlayers-{configs['model']['num_layers']}.pt"
    model.load_state_dict(
        torch.load(
            model_path,
            weights_only=True,
            map_location=map_location_dict,
        )
    )
    model.eval()
    start_time = time()
    main_jacobian(model, data, device, args)
    time_elapsed = round(time() - start_time, 1)
    print(f"Total Influence of {configs['method']} on {configs['dataset']} finished in {time_elapsed}s!")