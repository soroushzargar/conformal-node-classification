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
# endregion


class GraphSplit(object):
    def __init__(self, n_vertices, n_edges, edge_index, ys=None, device='cpu', undirected=True):
        self.device = device
        self.n_vertices = n_vertices
        self.n_edges = n_edges // 2 if undirected else n_edges

        self._vertices_budget = torch.ones(size=(self.n_vertices,), dtype=bool).to(self.device)
        self._edge_budget = torch.ones(size=(self.n_edges, ), dtype=bool).to(self.device)
        self.ys = ys
        self.stratified_enabled = False
        self._n_classes = None
        if not self.ys is None:
            self.stratified_enabled = True
            self._n_classes = ys.max() + 1
        if undirected:
            self.edge_index = self.onedir_edge_index(edge_index)
        else:
            self.edge_index = edge_index

    @staticmethod
    def onedir_edge_index(edge_index):
        e_filter = edge_index[0] < edge_index[1]
        return edge_index.T[e_filter].T

    def sample_nodes(self, n_nodes, return_mask=True, stratified=False):
        if stratified and (self.stratified_enabled == False):
            raise KeyError("Stratified sampling is not supported if n_classes is not mentioend")
        node_idxs = None
        if stratified:
            node_idxs = self._stratified_sample_nodes(n_nodes=n_nodes)
        else:
            node_idxs = self._overal_sample_nodes(n_nodes=n_nodes)

        node_mask = self._convert_to_mask(node_idxs, self.n_vertices)
        corresponding_edge_mask = self._compute_subgraph_edges(node_mask)
        self._vertices_budget = self._vertices_budget & (~node_mask)
        self._edge_budget = self._edge_budget & (~corresponding_edge_mask)

        if return_mask:
            return node_mask
        else:
            return node_mask.nonzero(as_tuple=True)[0]
    
    def sample_edges(self, n_edges, return_mask=False):
        edge_idxs = self._overal_sample_edges(n_edges=n_edges)
        edge_mask = self._convert_to_mask(edge_idxs, self.n_edges)

        # TODO: extract from nodes and edges
        used_vertices = torch.zeros_like(self._vertices_budget).to(self.device)
        used_vertices[self.edge_index_filter(self.edge_index, edge_mask).reshape(-1, )] = True

        self._vertices_budget = self._vertices_budget & (~used_vertices)
        self._edge_budget = self._edge_budget & (~edge_mask)

        if return_mask:
            return edge_mask
        else:
            return self.return_undirected_edges(edge_mask)
    
    def return_undirected_edges(self, edge_mask):
        e = self.edge_index_filter(self.edge_index, edge_mask)
        return torch_geometric.utils.to_undirected(e)
        
    @staticmethod
    def edge_index_filter(edge_index, edge_mask):
        return edge_index.T[edge_mask].T

    def compute_node_mask(self, edge_mask):
        used_vertices = torch.zeros_like(self._vertices_budget).to(self.device)
        used_vertices[self.edge_index_filter(self.edge_index, edge_mask).reshape(-1, )] = True
        return used_vertices
        
    def _overal_sample_edges(self, n_edges):
        target_edge_mask = self._edge_budget
        idxs = target_edge_mask.nonzero(as_tuple=True)[0][torch.randperm(target_edge_mask.sum())]
        return idxs[:n_edges]


    def _convert_to_mask(self, idx, n_elems):
        res = torch.zeros((n_elems, ), dtype=bool).to(self.device)
        res[idx] = True
        return res

    def _compute_subgraph_edges(self, node_mask):
        edge_mask = node_mask[self.edge_index[0]] & node_mask[self.edge_index[1]]
        return edge_mask

        
    def _overal_sample_nodes(self, n_nodes):
        target_node_mask = self._vertices_budget
        idxs = target_node_mask.nonzero(as_tuple=True)[0][torch.randperm(target_node_mask.sum()).to(self.device)]
        return idxs[:n_nodes]
        
        
    def _stratified_sample_nodes(self, n_nodes):
        result_idxs = []
        for class_i in range(self._n_classes):
            target_node_mask = ((self.ys == class_i) & (self._vertices_budget))
            idxs = target_node_mask.nonzero(as_tuple=True)[0][torch.randperm(target_node_mask.sum()).to(self.device)]
            result_idxs.append(idxs[:n_nodes])
        return torch.concat(result_idxs)

    @classmethod
    def from_dataset(cls, dataset):
        obj = cls(n_vertices=dataset.x.shape[0], n_edges=dataset.edge_index.shape[1], edge_index=dataset.edge_index, ys=dataset.y, device=dataset.x.device)
        # obj.device = dataset.x.device
        return obj
