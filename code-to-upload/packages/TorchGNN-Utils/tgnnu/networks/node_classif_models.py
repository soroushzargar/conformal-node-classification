import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GCNConv, GATConv, APPNP, SAGEConv
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Torch Graph Models are running on {device}")


class GCN(torch.nn.Module):
    def __init__(self, n_features, n_hidden, n_classes, p_dropout):
        super().__init__()
        self.conv1 = GCNConv(n_features, n_hidden)
        self.conv2 = GCNConv(n_hidden, n_classes)
        self.p_dropout = p_dropout

    def forward(self, X, edge_weight=None):
        x, edge_index = X.x, X.edge_index
        # x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x

class GAT(nn.Module):
    def __init__(self, n_features, n_hidden, n_classes, n_heads, p_dropout=0.6):
        super(GAT, self).__init__()
        self.conv1 = GATConv(n_features, n_hidden, heads=n_heads, concat=True)
        self.conv2 = GATConv(n_hidden * n_heads, n_classes, heads=1, concat=False)

        self.p_dropout = p_dropout

    def forward(self, X):
        node_attr, edge_index = X.x, X.edge_index
        hidden = self.conv1(node_attr, edge_index)
        hidden = F.elu(hidden)
        hidden = F.dropout(hidden, p=self.p_dropout, training=self.training)
        out = self.conv2(hidden, edge_index)

        return out

class APPNPNet(nn.Module):
    def __init__(self, n_features, n_hidden, n_classes, k_hops, alpha, p_dropout= 0.5):
        super(APPNPNet, self).__init__()
        self.lin1 = nn.Linear(n_features, n_hidden, bias=False)
        self.lin2 = nn.Linear(n_hidden, n_classes, bias=False)
        self.prop = APPNP(K = k_hops, alpha = alpha)
        self.p_dropout = p_dropout

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, X):
        node_attr, edge_index = X.x, X.edge_index

        hidden = F.relu(self.lin1(node_attr))
        hidden = F.dropout(hidden, p=self.p_dropout, training=self.training)
        out = self.lin2(hidden)
        out = self.prop(out, edge_index)
        return F.log_softmax(out, dim=1)

class SAGE(nn.Module):
    def __init__(self, n_features, n_hidden, n_classes, p_dropout= 0.5, aggregator='mean'):
        super(SAGE, self).__init__()

        self.conv1 = SAGEConv(n_features, n_hidden, normalize=False)
        self.conv1.aggr = aggregator
        self.transition = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(n_hidden),
            nn.Dropout(p=p_dropout)
        )
        self.conv2 = SAGEConv(n_hidden, n_classes, normalize=False)
        self.conv2.aggr = aggregator


    def forward(self,  X):
        node_attr, edge_index = X.x, X.edge_index
        hidden = self.conv1(node_attr, edge_index)
        hidden = self.transition(hidden)
        out = self.conv2(hidden, edge_index)
        return out

class MLP(nn.Module):
    def __init__(self, n_features, n_hidden, n_classes, p_dropout= 0.5):
        super(MLP, self).__init__()
        self.lin1 = nn.Linear(n_features, n_hidden)
        self.lin2 = nn.Linear(n_hidden, n_classes)

        self.p_dropout = p_dropout

    def forward(self, X):
        node_attr, edge_index = X.x, X.edge_index
        hidden = F.relu(self.lin1(node_attr))
        hidden = F.dropout(hidden, p=self.p_dropout, training=self.training)
        out = self.lin2(hidden)
        return out



