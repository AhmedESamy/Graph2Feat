import torch.nn.functional as F
import torch.nn as nn
import torch
import os
import pickle
from torch_geometric.nn import HeteroLinear, Linear, BatchNorm
from torch_geometric.datasets import Twitch, Planetoid, CitationFull, HGBDataset, IMDB
from torch_geometric.transforms import GDC, RandomLinkSplit
from copy import deepcopy
torch.manual_seed(10)


device = torch.device('cuda:1')

def save_file(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    
def open_file(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

class Student(nn.Module):
    
    def __init__(self, input_dim, output_dim, node_types, device, transductive_types = None):
        super().__init__()
        self.device = device
        self.em_dict = nn.ModuleDict()
        self.lin1 = nn.ModuleDict()
        self.bn1 = nn.ModuleDict()
        self.lin2 = nn.ModuleDict()
        self.bn2 = nn.ModuleDict()
        self.tt = transductive_types
        
        for nt in node_types:
            if transductive_types is not None and nt in transductive_types:
                self.em_dict[nt] = nn.Embedding(transductive_types[nt], 128) 
                nn.init.xavier_uniform_(self.em_dict[nt].weight)
            else:
                self.em_dict[nt] = None
                
            self.lin1[nt] = Linear(input_dim, output_dim)
            self.bn1[nt] = BatchNorm(output_dim)
            self.lin2[nt] = Linear(output_dim, output_dim)
            self.bn2[nt] = BatchNorm(output_dim)
                
    def forward(self, x_dict):
        for node_type in x_dict:
            if self.em_dict[node_type] is not None:
                x_dict[node_type] = self.em_dict[node_type](x_dict[node_type].to(self.device)).squeeze() 
               
            x_dict[node_type] = self.lin1[node_type](x_dict[node_type].to(self.device))
            x_dict[node_type] = self.bn1[node_type](x_dict[node_type])
            x_dict[node_type] = self.lin2[node_type](x_dict[node_type])
            x_dict[node_type] = self.bn2[node_type](x_dict[node_type])
            
        return x_dict

    
    
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


def e_loss(data, x, edgetypes):
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

        return loss.mean()



root = "./data/pakdd2023/"
name = "IMDB"

if name == "DBLP":
    dataset = HGBDataset(root+name, name)
    data = dataset.data
    print(data)
    data['venue'].x = torch.arange(data['venue'].num_nodes).reshape(-1, 1)
    targets = {"paper"}
    
elif name == "ACM":
    dataset = HGBDataset(root+name, name)
    data = dataset.data
    print(data)
    data['term'].x = torch.arange(data['term'].num_nodes).reshape(-1, 1)
    targets = {"paper"}

elif name == "IMDB":
    #dataset = IMDB(root+name)
    dataset = HGBDataset(root+name, name)
    data = dataset.data
    data['keyword'].x = torch.arange(data['keyword'].num_nodes).reshape(-1, 1)
    targets = {"movie"}   
    

if os.path.isfile(name+'_train_data.pickle'):

    train_data = open_file('datasplits/'+name+'_train_data.pickle')
    valid_data = open_file('datasplits/'+name+'_valid_data.pickle')
    test_data =  open_file('datasplits/'+name+'_test_data.pickle')
    
else:    
    rlp = RandomLinkSplit(
        edge_types=data.edge_types,
        neg_sampling_ratio=1.0,
    )

    train_data, valid_data, test_data = rlp(data)


    save_file(train_data, 'datasplits/'+name+'_train_data.pickle')
    save_file(valid_data, 'datasplits/'+name+'_valid_data.pickle')
    save_file(test_data, 'datasplits/'+name+'_test_data.pickle')
    
transductive_dict = None
if name == "DBLP":
    transductive_dict = {'venue': data['venue'].num_nodes}
elif name == "IMDB":
    transductive_dict = {'keyword': data['keyword'].num_nodes}
elif name == "ACM":
    transductive_dict = {'term': data['term'].num_nodes}
    
    

from TeacherHIN import Teacher
teacher = Teacher(train_data.metadata(), device = device, transductive_types = transductive_dict).to(device)

epochs = 1500
best = 0.0
optimizer = torch.optim.Adam(teacher.parameters(), lr=0.001)

for epoch in range(epochs):
     teacher.train()
     z = teacher(train_data)  
     loss = teacher.loss(train_data, z, batch_size=0)  
        
     optimizer.zero_grad()
     loss.backward()
     optimizer.step()
     
     #val_loss = teacher.train_step(x_=data.x, edge_index=train_data.edge_index, negatives=negatives, grad=False)
     teacher.eval()
     z = teacher(train_data)  
     auc, ap = evaluate(z, valid_data, valid_data.metadata()[1])
     
     if (epoch + 1) % 1 == 0:
         print(f"Epoch {epoch + 1}/{epochs}, "
               f"Training loss: {loss:.4f}, "
               f"AUC: {auc}, AP: {ap}")
     if ap > best:
        best = ap
        teacher.save(name+'_hin_model.pt')

teacher.load(name+'_hin_model.pt')
teacher.eval()

# z = teacher(train_data)  
# auc, ap = evaluate(z, valid_data, valid_data.metadata()[1])
# print(f"validation AUC: {auc}, AP: {ap}")



student = Student(-1, 128, train_data.node_types, device, transductive_dict).to(device)
optimizer2 = torch.optim.Adam(student.parameters(), lr=0.001)

epochs = 1000
losses = []
best_student = None
best = 0.0
for epoch in range(epochs):
    student_loss = 0.0
    student.train()
    student_x = student(train_data.x_dict)
    for nt in train_data.metadata()[0]:
        student_loss +=  F.mse_loss(student_x[nt], z[nt].detach())
    student_loss += e_loss(train_data, student_x, train_data.metadata()[1])
    optimizer2.zero_grad()
    student_loss.backward()
    optimizer2.step()
    losses.append(student_loss.item())

    with torch.no_grad():
        student.eval()
        student_x = student(valid_data.x_dict)
        val_loss = e_loss(valid_data, student_x, valid_data.metadata()[1])
        auc, ap = evaluate(student_x, valid_data, valid_data.metadata()[1])
        
    if ap > best :
        best = ap
        best_student = deepcopy(student)
        save_file(best_student, 'best_student.pickle')
        

    if (epoch + 1) % 1 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, "
          f"student loss: {student_loss.item():.8f}, "
          f"val loss: {val_loss.item():.8f}, "    
          f"validation AUC: {auc}, AP: {ap}")


            
# #testing       
with torch.no_grad():
    best_student = open_file('best_student.pickle')
    best_student.eval()
    student_x = best_student(test_data.x_dict)    
    auc, ap = evaluate(student_x, test_data, test_data.metadata()[1])
        
    print(f"testing AUC: {auc}, AP: {ap}")