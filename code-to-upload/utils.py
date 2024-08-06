# region imports
import os
import sys
import math
from abc import ABC
import copy

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as TVDatasets

import torch_geometric
from torch_geometric.data import Data as GraphData
from torch_geometric.nn import GCNConv
from torch_geometric.nn.models.label_prop import LabelPropagation
import torch_geometric.datasets as pyg_datasets
from torch_geometric.utils import to_networkx

import seaborn as sns
import matplotlib.pyplot as plt
device = 'cuda' if torch.cuda.is_available() else 'cpu'

from tqdm.notebook import tqdm

import sys
#endregion

# node-induced subgraph
def node_induced_subgraph(graph, mask):
    new_edge_index = graph.edge_index.T[
        mask[graph.edge_index[0]] & mask[graph.edge_index[1]]
        ].T.clone()
    
    return GraphData(x=graph.x, edge_index=new_edge_index, y=graph.y)

# edge-induced subgraph
def edge_induced_subgraph(graph, edge_mask):
    new_edge_index = graph.edge_index.T[edge_mask].T.clone()
    return GraphData(x=graph.x, edge_index=new_edge_index, y=graph.y)

# Union of two edges
def union_edge_index(graph, first, second):
    edge_index = torch_geometric.utils.sort_edge_index(torch.concat([first, second], dim=1))
    return GraphData(x=graph.x, edge_index=edge_index, y=graph.y)


# converts edge_index to True/False node_mask
def edges_to_node_mask(edge_index, n_vertices):
    node_index = edge_index.reshape(-1, ).unique()
    node_mask = torch.zeros((n_vertices, ), dtype=bool)
    node_mask[node_index] = True
    return node_mask.to(edge_index.device)