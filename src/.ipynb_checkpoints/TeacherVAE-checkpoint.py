from torch_geometric.nn.models import DeepGraphInfomax
from torch_geometric.nn import SAGEConv, GCNConv

import torch.nn.functional as F
import torch.nn as nn
import torch

import os.path as osp

from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import DataLoader
from torch import nn, Tensor
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, HeteroConv, Linear, GCNConv, GAE, VGAE
from torch_scatter import scatter
from torch.nn import BatchNorm1d, ReLU, Sequential
from torch_geometric.nn import global_mean_pool, global_add_pool

import torch
import torch.nn.functional as F
import numpy as np
import math


torch.manual_seed(1)
torch.cuda.manual_seed_all(1)


class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, out_channels)
        self.conv2 = SAGEConv(out_channels, out_channels)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return  x


class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = GCNConv(in_channels, out_channels)
        self.conv_mu =GCNConv(in_channels, out_channels)
        self.conv_logstd = GCNConv(in_channels, out_channels)
        self.prelu = nn.PReLU()
        
    def forward(self, x, edge_index):
        #x = F.rrelu(self.conv(x, edge_index))
        mu = (self.conv_mu(x, edge_index))
        log_std = (self.conv_logstd(x, edge_index))
        return mu, log_std
    


class Teacher(nn.Module):
    def __init__(self, input_dim, output_dim=128, device= 'cpu'):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.encoder =  VGAE(VariationalGCNEncoder(input_dim, output_dim))
        #self.encoder = GAE(GCNEncoder(input_dim, output_dim))
        
    
    def forward(self, data):
        z = self.encoder.encode(data.x, data.edge_index)
        return z
    
    
    def loss(self, data, z: torch.Tensor, variational=True):
        loss = self.encoder.recon_loss(z, data.edge_index)
        if variational:
            loss = loss + (1 / data.num_nodes) * self.encoder.kl_loss()
            
        return loss
    
    
    def load(self, path="hom_model.pt"):
        self.load_state_dict(torch.load(path))

    def save(self, path="hom_model.pt"):
        torch.save(self.state_dict(), path)

        
    
