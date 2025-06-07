# City-Networks
We introduce [***City-Networks***](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.CityNetwork.html?highlight=city#torch_geometric.datasets.CityNetwork), a transductive learning dataset for testing long-range dependencies in Graph Neural Networks (GNNs).
In particular, our dataset contains four large-scale city maps: Paris, Shanghai, L.A., and London, where nodes represent intersections and edges represent road segments.

At the same time, we introduce [***Total Influence***](https://pytorch-geometric.readthedocs.io/en/latest/modules/utils.html#torch_geometric.utils.total_influence), a measurement based on the Jacobians that quantifies long-range dependency of a trained GNN model for node-level tasks.

**Paper: [Towards Quantifying Long-Range Interactions in Graph Machine Learning: a Large Graph Dataset and a Measurement
](https://arxiv.org/abs/2503.09008).**

**Update:** [***CityNetwork***](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.CityNetwork.html?highlight=city#torch_geometric.datasets.CityNetwork) and [***Total Influence***](https://pytorch-geometric.readthedocs.io/en/latest/modules/utils.html#torch_geometric.utils.total_influence) are now both available in the latest version of [**Pytorch Geometric**](https://pytorch-geometric.readthedocs.io/en/latest/index.html) (2.7.0) 🚀

<div align="center">
  <img src="Figures/road_networks_visualizations_cities.jpg" alt="cities" style="width: 99%; height: 99%">
</div>


## Visualization
The nodes are labeled by an approximation of eccentricity, which measures the accessibility of a node in the network. We visualize two sub-regions in our dataset below, 
where darker color indicates lower node eccentricity (i.e. more accessible). 

<div align="center">
  <img src="Figures/label_visual_map_huangpu_and_pasadena_16-hop_10-chunk.jpg" alt="labels" style="width: 99%; height: 99%">
</div>

A more detailed discussion can be found in the [paper](https://arxiv.org/abs/2503.09008).


## Load CityNetwork as a PyG Dataset
You can easily use the `CityNetwork` class from `citynetworks.py` to load our dataset as a PyG InMemory Dataset:
```python
from citynetworks import CityNetwork
```
You can also load [***CityNetwork***](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.CityNetwork.html?highlight=city#torch_geometric.datasets.CityNetwork) from the latest [Pytorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/index.html) (2.7.0) by installing its [**nightly version**](https://pypi.org/project/pyg-nightly/):
```bash
pip install pyg-nightly
```
or from [**master**](https://github.com/pyg-team/pytorch_geometric):
```bash
pip install git+https://github.com/pyg-team/pytorch_geometric.git
```
**Example usage**:
```python
from torch_geometric.datasets import CityNetwork

dataset = CityNetwork(root="./city_networks", name="paris")
paris_network = dataset[0]
```
Here `name` takes values in `[paris, shanghai, la, london]`.

## Total Influence Calculation
Given a trained GNN `model` with its underlying graph `data` from PyG, the [***Total Influence***](https://pytorch-geometric.readthedocs.io/en/latest/modules/utils.html#torch_geometric.utils.total_influence) can be calculated with the `total_influence` method from `influence/influence.py`:
```python
from influence import total_influence
```
or from the latest [Pytorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/index.html) (2.7.0):
```python
from torch_geometric.utils import total_influence

model.eval()

avg_tot_inf, R = total_influence(
    model, 
    data, 
    max_hops=16, # the largest hop to consider
    num_samples=10000, # number of node samples
    normalize=True, # if to normalize the influence by hop-0
    average=True, # if to return the averaged total influence
    device='cuda:0', # cuda or cpu
    vectorize=True, # vectorize in torch.autograd.functional.jacobian
)
```
Here `avg_tot_inf` is the averaged total influence at each hop, and `R` is the breadth of influence-weighted receptive field averaged over the node samples. 

**Note:** although the scope of our paper is under transductive settings, this measurement also works for inductive node-level tasks. 

Readers may refer to Section 4 of our paper for a detailed discussion of this measurement.

## Baseline Results

### 1. Empirical Performance
We test several standard GNNs and a Graph Transformer on our city networks with a *train/valid/test* split of *10%/10%/80%*, and then monitor their behaviors at different layers. 

In particular, we consider #hops = #layers = [2, 4, 8, 16]. The results below suggest a clear gain in performance by increasing the number of layers on our city networks, as opposed to Cora where the models suffer from over-smoothing problems.

<div align="center">
  <img src="Figures/baseline_results.jpg" alt="labels" style="width: 99%; height: 99%">
</div>

### 2. Per-hop Influence
We further show the per-hop influence (measured by the Jacobian) under \#layers = 16. We can observe from the following results that the influence from distant nodes decays at a much slower rate on our city networks compared to the rate on other social networks.

<div align="center">
  <img src="Figures/influence_results.jpg" alt="labels" style="width: 99%; height: 99%">
</div>

A more detailed discussion can be found in the [paper](https://arxiv.org/abs/2503.09008).


## Benchmarking

### 1. Create a Virtual Environment for CityNetworks
```bash
conda create -n citynetwork python=3.10
conda activate citynetwork
```

### 2. Install Packages
Simply use `install.sh` to install the required dependencies. The detailed package versions are specified in `requirements.txt`.
```bash
bash install.sh
```

### 3. Run the Training Pipeline
Use `experiments_run.sh` to run the baselines on our City-Networks.
```bash
bash train_run.sh
```
The results will be saved under `./results/` and the model checkpoints will be saved under `./models/`.

### 4. Calculate the Total Influence
Use `influence_run.sh` to calculate the total influence based on the saved model checkpoint.
```bash
bash influence_run.sh
```
The results will be saved under `./influence_results/`.

### 5. Visualize the Baseline Resutls & Influence
**You can easily plot the baseline results and influence scores using `plot_results.ipynb`.**


## Reproduce the Dataset

Inside the virtual environment, use the commands in a bash shell to generate city networks with pre-specified queries in `dataset_generation/places.py`,
```bash
cd dataset_generation

python generate_network.py --place paris
python generate_network.py --place shanghai
python generate_network.py --place la
python generate_network.py --place london
```
You can also visualize the city networks and their annotations using `Visualize_CityNetworks.ipynb` and `Visualize_Eccentricity.ipynb`.

## Citation
Please cite our paper if you find the repo helpful in your work.
```bibTex
@article{liang2025towards,
  title={Towards Quantifying Long-Range Interactions in Graph Machine Learning: a Large Graph Dataset and a Measurement},
  author={Liang, Huidong and Borde, Haitz S{\'a}ez de Oc{\'a}riz and Sripathmanathan, Baskaran and Bronstein, Michael and Dong, Xiaowen},
  journal={arXiv preprint arXiv:2503.09008},
  year={2025}
}
```
