from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch_geometric.transforms import AddLaplacianEigenvectorPE, AddRandomWalkPE, GDC
from torch.utils.data import DataLoader
from torch import nn, Tensor
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, HeteroConv, Linear, GCNConv
from torch_scatter import scatter
from torch_geometric.nn import HeteroLinear, Linear, BatchNorm

import torch
import torch.nn.functional as F
import numpy as np
import math


torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
    
def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    
class Encoder(nn.Module):
    def __init__(self, metadata, hidden_channels, num_layers = 1, transductive_types = None):
        super().__init__()
        self.conv = HeteroConv({edge_type: SAGEConv((-1, -1),  hidden_channels) for edge_type in metadata[1]}, aggr = 'mean')
        self.lin = nn.ModuleDict()
        self.relu = torch.nn.PReLU()
        self.em_dict = nn.ModuleDict()
        self.bn = nn.ModuleDict()
              
        for nt in metadata[0]:
            if transductive_types is not None and nt in transductive_types:
                self.em_dict[nt] = nn.Embedding(transductive_types[nt], 128) 
                nn.init.xavier_uniform_(self.em_dict[nt].weight)         
        
        
    def forward(self, x_dict, edge_index, infer = False):
        device = next(self.parameters()).device
        
        def to_device(x_dict): 
            return {k: v.to(device) for k, v in x_dict.items()}
        
        for node_type in x_dict:
            if node_type in self.em_dict:
                x_dict[node_type] = self.em_dict[node_type](x_dict[node_type].to(device)).squeeze() 
        
        x = self.conv(to_device(x_dict), edge_index_dict=to_device(edge_index))
        return x
   


class Teacher(nn.Module):
    def __init__(self, metadata, transductive_types, output_dim=128, lr=0.1, device= 'cpu'):
        super().__init__()
        self.metadata = metadata
        self.output_dim = output_dim
        self.lr = lr
        self.device = device
        self.encoder = Encoder(metadata, output_dim, transductive_types = transductive_types)
        
    
    def forward(self, data):
        return self.encoder(data.x_dict, data.edge_index_dict)
            
    
    def e_loss(self, data, x):
        r"""Computes the loss given positive and negative random walks."""
        loss = 0.0
        
        for edge_type in self.metadata[1]:
            k = list(x.keys())[0]
            edge_index = data[edge_type].edge_label_index.to(self.device)
            labels = data[edge_type].edge_label.long().to(self.device)
        
            # Positive loss.
            EPS = 0.0000001
            src, trg = edge_index
            src_type, trg_type = edge_type[0], edge_type[2]
        

            src_x = x[src_type][src][labels.bool()]
            trg_x = x[trg_type][trg][labels.bool()]

            h_start = src_x
            h_rest = trg_x

            out = (h_start * h_rest).sum(dim=-1).view(-1)
            pos_loss = -torch.log(torch.sigmoid(out) + EPS).mean()

            # Negative loss.
            src_x = x[src_type][src][~labels.bool()]
            trg_x = x[trg_type][trg][~labels.bool()]

            h_start = src_x
            h_rest = trg_x

            out = (h_start * h_rest).sum(dim=-1).view(-1)
            neg_loss = -torch.log(1 - torch.sigmoid(out) + EPS).mean()

            loss += pos_loss + neg_loss
            
            del edge_index, labels

        return loss.mean()
    
    def loss(self, data, z_dict: torch.Tensor,
             mean: bool = True, batch_size: int = 0):
                
        e_loss = self.e_loss(data, z_dict)
        return e_loss
    
    
    def load(self, path="model.pt"):
        self.load_state_dict(torch.load(path))

    def save(self, path="model.pt"):
        torch.save(self.state_dict(), path)

        
        
