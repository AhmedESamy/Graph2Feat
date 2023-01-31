import torch.nn.functional as F
import torch.nn as nn
import torch
from torch_geometric.nn import HeteroLinear, Linear, BatchNorm

from torch_geometric.datasets import Twitch, CitationFull, HGBDataset, IMDB, DBLP
from torch_geometric.transforms import GDC, RandomLinkSplit
from torch_geometric.utils import structured_negative_sampling
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.utils import dropout_adj
torch.manual_seed(10)


root = "./temp"
name = "DBLP"

if name == "DBLP":
    dataset = CitationFull(root+name, name)
    data = dataset.data
    
print(data)     
device = torch.device('cuda:1')


class HetLinear(torch.nn.Module):
    def __init__(self, input_features = -1, out_dim = 128, node_types = 4):
        super().__init__()
        self.lin = nn.ModuleList()
        for _ in range(node_types):
            self.lin.append(Linear(input_features, out_dim))

    def forward(self, x):
        for (i, (k, v)) in enumerate(x.items()):
            if i < len(self.lin)-1: 
                x[k] = self.lin[i](v)
            else: 
                x[k] = self.lin[i](v)
        return x
    

class Student(torch.nn.Module):
    def __init__(self, node_types = 4):
        super().__init__()
        self.lin1 = HetLinear()
        self.batch1 = BatchNorm(128)
        self.lin2 = HetLinear(input_features=128, out_dim=128)
        self.batch2 = BatchNorm(128)

    def forward(self, x_dict):
        x = self.lin1(x_dict)
        for (k, v) in x.items():
            x[k] = self.batch1(v)
        x = self.lin2(x)
        for (k, v) in x.items():
            x[k] = self.batch2(v)
        return x


def drop_feature(x, drop_prob= 0.5):
        drop_mask = torch.empty(
            (x.size(1),),
            dtype=torch.float32, ).uniform_(0, 1) < drop_prob
        x = x.clone()
        x[:, drop_mask] = 0
        return x
    
    
def evaluate(x, data, edge_types):
    from sklearn.metrics import roc_auc_score, average_precision_score
    auc = 0; ap = 0
    for edge_type in edge_types:
            
        edge_index = data[edge_type].edge_label_index
        labels = data[edge_type].edge_label.long()
        
        s, t = edge_index
        src_type, trg_type = edge_type[0], edge_type[2]
    
        s_emb = x[src_type][s].detach()
        t_emb = x[trg_type][t].detach()
        scores = s_emb.mul(t_emb).sum(dim=-1).cpu().numpy()
        auc += roc_auc_score(y_true=labels, y_score=scores)
        ap += average_precision_score(y_true=labels, y_score=scores)
        
    return auc/len(edge_types), ap/len(edge_types)


def e_loss(data, x, edgetypes, device):
        r"""Computes the loss given positive and negative random walks."""
        loss = 0.0
        for edge_type in edgetypes:
            
            edge_index = data[edge_type].edge_label_index
            labels = data[edge_type].edge_label.long()
        
            # Positive loss.
            EPS = 0.0000001
            src, trg = edge_index
            src_type, trg_type = edge_type[0], edge_type[2]
        

            src_x = x[src_type][src][labels.bool()].to(device)
            trg_x = x[trg_type][trg][labels.bool()].to(device)

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

        return loss
        
    
def generate_augmented_views(data, adj_dropout1=0.2, adj_dropout2=0.4, f_dropout1=0.3, f_dropout2=0.4):
    aug_data1 = data.clone()
    aug_data2 = data.clone()
    
    for edge_type in data.metadata()[1]:
        aug_data1[edge_type].edge_label_index = dropout_adj(aug_data1[edge_type].edge_label_index, p=adj_dropout1)[0]
        aug_data2[edge_type].edge_label_index = dropout_adj(aug_data2[edge_type].edge_label_index, p=adj_dropout2)[0]

    for node_type in data.metadata()[0]:
        aug_data1[node_type].x = drop_feature(aug_data1[node_type].x, f_dropout1)
        aug_data2[node_type].x = drop_feature(aug_data2[node_type].x, f_dropout2)
                
    return aug_data1, aug_data2
 


