# City-Networks
We introduce **City-Networks**, a transductive learning dataset for testing long-range dependencies in Graph Neural Networks (GNNs).
In particular, our dataset contains four large-scale city maps: Paris, Shanghai, L.A., and London, where nodes represent intersections and edges represent road segments.

**Note**: We are currently integrating our dataset into [Pytorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/index.html) and expect to release it very soon.

Paper: [Towards Quantifying Long-Range Interactions in Graph Machine Learning: a Large Graph Dataset and a Measurement
](https://arxiv.org/abs/2503.09008).

<div align="center">
  <img src="Figures/road_networks_visualizations_cities.jpg" alt="cities" style="width: 99%; height: 99%">
</div>


## Visualization
The nodes are labeled by an approximation of eccentricity, which measures the accessibility of a node in the network. We visualize two sub-regions in our dataset below, 
where darker color indicates lower node eccentricity (i.e. more accessible). 

<div align="center">
  <img src="Figures/label_visual_map_huangpu_and_pasadena_16-hop_10-chunk.jpg" alt="labels" style="width: 99%; height: 99%">
</div>

Please refer to our paper for a more detailed discussion.

## Installation
### 1. Create a virtual environment for GraphComBO
```bash
conda create -n citynetworks python=3.10
conda activate citynetworks
```

### 2. Install Packages
```bash
pip install networkx
pip install torch_geometric
pip install osmnx
pip install publib
```

## Load as PyG Datasets

**ToDo**

We are currently integrating our dataset into [Pytorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/index.html) and expect to release it very soon.


## Reproduce the Networks
Use the commands in a bash shell to generate city networks with pre-specified queries in `places.py`,
```bash
python generate_network.py --place paris
python generate_network.py --place shanghai
python generate_network.py --place la
python generate_network.py --place london
```

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
