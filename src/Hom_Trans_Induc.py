import torch.nn.functional as F
import torch.nn as nn
import torch
import pickle
import os
from torch_geometric.datasets import Twitch, AttributedGraphDataset, CitationFull, WikipediaNetwork, Planetoid, Coauthor
from torch_geometric.transforms import GDC, RandomLinkSplit
from torch_geometric.utils import structured_negative_sampling
from torch_geometric.data import Data
from torch_geometric.utils import dropout_adj, to_networkx

from copy import deepcopy, copy
from torch_geometric.utils import subgraph

torch.manual_seed(10)
device = "cuda:1"

    
def save_file(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    
def open_file(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def evaluate(x, data):
    from sklearn.metrics import roc_auc_score, average_precision_score
    edge_index = data.edge_label_index
    labels = data.edge_label.long().cpu()
    
    s, t = edge_index
    s_emb = x[s].detach().cpu()
    t_emb = x[t].detach().cpu()

    scores = s_emb.mul(t_emb).sum(dim=-1)
    auc = roc_auc_score(y_true=labels, y_score=scores)
    ap = average_precision_score(y_true=labels, y_score=scores)
    return auc, ap


def e_loss(x, data):
    r"""Computes the loss given positive and negative random walks."""
    edge_index = data.edge_label_index
    labels = data.edge_label.long()

    # Positive loss.
    EPS = 0.0000001
    src, trg = edge_index

    src_x = x[src][labels.bool()]
    trg_x = x[trg][labels.bool()]

    h_start = src_x
    h_rest = trg_x

    out = (h_start * h_rest).sum(dim=-1).view(-1)
    pos_loss = -torch.log(torch.sigmoid(out) + EPS).mean()

    # Negative loss.
    src_x = x[src][~labels.bool()]
    trg_x = x[trg][~labels.bool()]

    h_start = src_x
    h_rest = trg_x

    out = (h_start * h_rest).sum(dim=-1).view(-1)
    neg_loss = -torch.log(1 - torch.sigmoid(out) + EPS).mean()

    loss = pos_loss + neg_loss
    return loss




#train_data, valid_data, test_data = torch.load("split.pt")
root = "./data/pakdd2023/"
name = "Wikipedia"
# print(name)
# dataset = WikipediaNetwork(root+name, "chameleon")
# data = dataset.data
# print(data)
# data.x = data.x.to_dense()

# print(data)


# import networkx   
# from torch_geometric.utils import to_networkx, from_networkx
# G = to_networkx(data, to_undirected=True, node_attrs = ["x"])
# Gcc = max(networkx.connected_components(G), key=len)

# dict_ = {}
# for g in Gcc:
#     length = len(G.subgraph(g).nodes)
#     if length not in dict_.keys():
#         dict_[length] = 1
#     else:
#         dict_[length] +=1

# print(dict_, "num_nodes", data.num_nodes)

# data = from_networkx(G.subgraph(Gcc))
# print(data)

inductive = False

if inductive:
    
    if os.path.isfile('datasplits/'+'ind'+name+'_train_data.pickle'):
        train_data = open_file('datasplits/'+'ind'+name+'_train_data.pickle')
        train_train_data = open_file('datasplits/'+'ind'+name+'_train_train_data.pickle')
        train_valid_data = open_file('datasplits/'+'ind'+name+'_train_valid_data.pickle')
        valid_data = open_file('datasplits/'+'ind'+name+'_valid_data.pickle')
        test_data = open_file('datasplits/'+'ind'+name+'_test_data.pickle')
    
    else:
        rands = torch.rand(data.num_nodes)
        train_mask =  rands < 0.3
        test_mask = rands > 0.65
        
        val_mask = []
        for i in torch.arange(data.num_nodes): 
            if (i not in train_mask.nonzero() and i not in test_mask.nonzero()):
                val_mask.append(True)
            else:
                val_mask.append(False)
        val_mask  = torch.Tensor(val_mask).to(torch.bool)
        
        train_data = copy(data)
        train_data.edge_index, _ = subgraph(train_mask, data.edge_index, relabel_nodes=True)
        train_data.x = data.x[train_mask]

        val_data = copy(data)
        val_data.edge_index, _ = subgraph(val_mask, data.edge_index, relabel_nodes=True)
        val_data.x = data.x[val_mask]
        
        test_data = copy(data)
        test_data.edge_index, _ = subgraph(test_mask, data.edge_index, relabel_nodes=True)
        test_data.x = data.x[test_mask]
        
#         G = to_networkx(train_data, to_undirected=True, node_attrs = ["x"])
#         Gcc = sorted(networkx.connected_components(G), key=len, reverse=True)
#         train_data = from_networkx(G.subgraph(Gcc[0]))
        
#         G = to_networkx(val_data, to_undirected=True, node_attrs = ["x"])
#         Gcc = sorted(networkx.connected_components(G), key=len, reverse=True)
#         val_data = from_networkx(G.subgraph(Gcc[0]))
        

        # For teacher model, split the training graph data for transductive setting
        lsp_transform = RandomLinkSplit(num_val=0.1, num_test=0)
        train_train_data, train_valid_data , _ = lsp_transform(
            Data(
                x = train_data.x,
                edge_index=train_data.edge_index,
                num_nodes=train_data.num_nodes
            )
        )

        print(train_data)
        lsp_transform = RandomLinkSplit(num_val=0.0, num_test=0)
        train_data, _, _ = lsp_transform(
            Data(
                x = train_data.x,
                edge_index=train_data.edge_index,
                num_nodes=train_data.num_nodes
            )
        )
        print(train_data)
        valid_data, _, _ = lsp_transform(
            Data(
                x = val_data.x,
                edge_index=val_data.edge_index,
                num_nodes=val_data.num_nodes
            )
        )
        
        test_data, _, _ = lsp_transform(
            Data(
                x = test_data.x,
                edge_index=test_data.edge_index,
                num_nodes=test_data.num_nodes
            )
        )
        
        
        save_file(train_data, 'datasplits/'+'ind'+name+'_train_data.pickle')
        save_file(train_train_data,'datasplits/'+ 'ind'+name+'_train_train_data.pickle')
        save_file(train_valid_data, 'datasplits/'+'ind'+name+'_train_valid_data.pickle')
        save_file(valid_data, 'datasplits/'+'ind'+name+'_valid_data.pickle')
        save_file(test_data, 'datasplits/'+'ind'+name+'_test_data.pickle')


else:
    
     
    if os.path.isfile('datasplits/'+name+'_train_data.pickle'):
   
        train_data = open_file('datasplits/'+name+'_train_data.pickle')
        valid_data = open_file('datasplits/'+name+'_valid_data.pickle')
        test_data = open_file('datasplits/'+name+'_test_data.pickle')
         
    else:    
        lsp_transform = RandomLinkSplit(num_val = 0.45, num_test = 0.45)
        train_data, valid_data, test_data = lsp_transform(
            Data(
                x = data.x,
                edge_index=data.edge_index,
                num_nodes=data.num_nodes
            )
        )

        save_file(train_data, 'datasplits/'+name+'_train_data.pickle')
        save_file(valid_data, 'datasplits/'+name+'_valid_data.pickle')
        save_file(test_data, 'datasplits/'+name+'_test_data.pickle')


from TeacherVAE import Teacher
if inductive:
    torch.manual_seed(10) 
    teacher = Teacher(input_dim=train_train_data.num_features, output_dim = 256, device=device).to(device)
else:
    teacher = Teacher(input_dim=train_data.num_features, device=device).to(device)
optimizer = torch.optim.Adam(teacher.parameters(), lr=0.001)

epochs = 1000
best = 0.0

for epoch in range(epochs):
    teacher.train()
    if inductive:
        z = teacher(train_train_data.to(device))  
        loss = teacher.loss(train_train_data.to(device), z)  
    else:         
        z = teacher(train_data.to(device))  
        loss = teacher.loss(train_data.to(device), z)  
        
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    teacher.eval()
    
    with torch.no_grad():
        if inductive:
            z = teacher(train_train_data.to(device))  
            auc, ap = evaluate(z, train_valid_data)
        else:
            z = teacher(valid_data.to(device))  
            auc, ap = evaluate(z, valid_data)
    
    if (epoch + 1) % 10 == 0:
         print(f"Epoch {epoch + 1}/{epochs}, "
               f"Training loss: {loss:.4f}, "
               f"AUC: {auc}, AP: {ap}")
            
    if ap > best:
        best = ap
        teacher.save(name+'_hom_model.pt') 

        
teacher.load(name+'_hom_model.pt')
teacher.eval()

z = teacher(train_data.to(device))

if inductive:
    auc, ap = evaluate(z, train_valid_data)
else:
    auc, ap = evaluate(z, valid_data)
print(f"validation AUC: {auc}, AP: {ap}")

    
class Student(nn.Module):
    def __init__(self, input_dim, emb_dim, device):
        super(Student, self).__init__()
        self.device = device
        self.mlp = nn.Sequential(
          nn.Linear(input_dim, emb_dim),
          nn.Linear(emb_dim, emb_dim)
        )
        
        
    def forward(self, x):
        #x = F.rrelu(self.Linear1(x.to(device)))
        #x = F.rrelu(self.Linear2(x))
        return self.mlp(x)     
    
    
    
    
class indStudent(nn.Module):
    def __init__(self, input_dim, emb_dim, device):
        super(indStudent, self).__init__()
        self.device = device
        self.Linear1 = nn.Linear(input_dim, emb_dim).to(self.device)
        self.Linear2 = nn.Linear(emb_dim, emb_dim).to(self.device)
        #self.prelu = nn.PReLU()
        
    def forward(self, x, z = None):
        x = F.rrelu(self.Linear1(x.to(device)))
        #x = F.rrelu(self.Linear2(x.to(device)))
        # if z is not None:
        #      x = F.rrelu(torch.add(x, z)/2)
        return x               
    
    
torch.manual_seed(10) 
if inductive:
    student = indStudent(train_data.num_features, 128, device).to(device)
else:
    student = nn.Sequential(
          nn.Linear(train_data.num_features, 5 * 128),
          nn.RReLU(),
          nn.Linear(5* 128, 128),
          nn.RReLU()).to(device)


distill = True
#optimizer2 = torch.optim.Adam(student.parameters(), lr=0.001)
epochs =1000
losses = []
best_student = None
best = 0.0
beta = [0.005]
gamma = [1]
for b in beta:
    for q in gamma:
        
        if inductive:
            student = indStudent(train_data.num_features, 256, device).to(device)
        else:
            student = Student(train_data.num_features, 128, device).to(device)
       
        optimizer2 = torch.optim.Adam(student.parameters(), lr=0.001)
 
        best = 0.0
        best_student = None
        losses = []
   
        print(q, b)
    
        for epoch in range(epochs):
            torch.manual_seed(10)
            student.train()
            loss = 0
            student_x = student(train_data.x.to(device))
            if distill:
                loss += q * F.mse_loss(student_x, z.detach())   
            loss +=  b * e_loss(student_x, train_data)
            optimizer2.zero_grad()
            loss.backward()
            optimizer2.step()
            losses.append(loss.item())

            with torch.no_grad():
                student.eval()
                x = student(valid_data.x.to(device))
                val_loss = e_loss(x, valid_data)
                auc, ap = evaluate(x, valid_data)

            if ap > best :
                best = ap
                best_student = deepcopy(student)
                save_file(best_student, 'best_student.pickle')

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, "
                  f"student loss: {loss.item():.8f}, "
                  f"val loss: {val_loss.item():.8f}, "    
                  f"validation AUC: {auc}, AP: {ap}")

        import time

        # #testing       
        with torch.no_grad():
            best_student = open_file('best_student.pickle')
            best_student.eval()
            test_data = test_data.to(device)
            t_0 = time.time()
            student_x = best_student(test_data.x)    
            t_1 = time.time()
            elapsed_time = round((t_1 - t_0) * 10 ** 3, 3)
            print(elapsed_time)
            auc, ap = evaluate(student_x, test_data)

            print(f"testing AUC: {auc}, AP: {ap}")