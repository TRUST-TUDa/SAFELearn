import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import sys
import sklearn.metrics as sklm
import os
from torcheval.metrics.functional import multiclass_accuracy, multiclass_f1_score, multiclass_precision, multiclass_auroc

torch.manual_seed(42)

# Determine device (for GPU support)
device = torch.device("cpu")
#if torch.cuda.is_available():
#    device = torch.device("cuda")
#    torch.cuda.manual_seed(42)
#    torch.cuda.manual_seed_all(42)

print("Device:", device)
fullset = pd.read_csv('data/PPMI/PPMI_cleaned_altered.csv')
fullset = torch.Tensor(fullset.to_numpy())

generator = torch.Generator().manual_seed(42)
torch.manual_seed(42)

set_size = len(fullset)
clients = []
number_of_clients =1

# Split the data into 100 non-overlapping parts
split_size = len(fullset) // number_of_clients
for i in range(number_of_clients):
    split = fullset[i * split_size: (i + 1) * split_size]
    clients.append(split)

n_epochs = 10000
batch_size = 64

class DiabModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(12,20)
        self.linear2 = nn.Linear(20,15)
        self.linear3 = nn.Linear(15,12)
        self.linear4 = nn.Linear(12,8)
        self.linear5 = nn.Linear(8,4)
        self.linear6 = nn.Linear(4,3)
        self.act_fn = nn.SiLU()
        self.sigm = nn.Sigmoid()


    def forward(self, x):
        x = self.linear1(x)
        x = self.act_fn(x)
        x = self.linear2(x)
        x = self.act_fn(x)
        x = self.linear3(x)
        x = self.act_fn(x)
        x = self.linear4(x)
        x = self.act_fn(x)
        x = self.linear5(x)
        x = self.act_fn(x)
        x = self.linear6(x)
        #x = self.sigm(x)
        return x



def eval_model(model, X_test, y_test):
    model.eval()

    with torch.no_grad():
        y_pred = model(X_test)
        y_pred_integer = y_pred.round().cpu().numpy()
        print(multiclass_f1_score(y_pred, torch.reshape( y_test, (-1, )), num_classes=3))
        print(multiclass_auroc(y_pred, torch.reshape( y_test, (-1, )), num_classes=3))
        #precision_sklm = sklm.precision_score(y_test, y_pred_integer, average ='samples', labels= ['1, 2, 4'])
        #precision_sklm_inv = sklm.precision_score(y_pred_integer, y_test, average='samples')
        #recall_sklm = sklm.recall_score(y_test, y_pred_integer, average='samples')
        #recall_sklm_inv = sklm.recall_score(y_pred_integer,y_test, average='samples')
        #f1score_sklm = sklm.f1_score(y_test, y_pred_integer, average='samples')
        #f1score_sklm = sklm.f1_score(y_pred_integer,y_test, average='samples')
        #accuracy_sklm = sklm.accuracy_score(y_test, y_pred_integer, average='samples')
#
        #print("precision: ", precision_sklm)
        #print("precision inv:", precision_sklm_inv)
        #print("recall: ", recall_sklm)
        #print("recall: inv", recall_sklm_inv)
        #print("f1-score: ", f1score_sklm)
        #print("accuracy: ", accuracy_sklm)

# if the global model does not yet exist create a new fully untrained one
#if not os.path.exists("model/PPMImodels/GlobalModel"):
#    model = DiabModel()
#    torch.save(model.state_dict(), f"model/PPMImodels/GlobalModel") 


if (len(sys.argv) == 1):
    for split_index, split_data in enumerate(clients):
        train_size = int(0.8 * len(split_data))
        test_size = len(split_data) - train_size
        train_data, test_data = data.random_split(split_data, [train_size, test_size], generator=generator)
        
        X_train = torch.tensor(train_data.dataset[:, 2:], dtype=torch.float32)
        y_train = torch.tensor(train_data.dataset[:, 1], dtype=torch.float32).reshape(-1, 1)
        X_test = torch.tensor(test_data.dataset[:, 2:], dtype=torch.float32)
        y_test = torch.tensor(test_data.dataset[:, 1], dtype=torch.float32).reshape(-1, 1)
        
        model = DiabModel()
        # if there exists a global model from earlier learnings import it

        #model.load_state_dict(torch.load("model/PPMImodels/GlobalModel"))
        
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters())
        def train_model(model, optimizer, X_train, y_train, loss_fn, n_epochs=100):
            model.train()
            model.to(device)
            for epoch in range(n_epochs):
                for i in range(0, len(X_train), batch_size):
                    Xbatch = X_train[i:i+batch_size].to(device)
                    ybatch = y_train[i:i+batch_size].to(device)
                    y_pred = model(Xbatch)
                    loss = loss_fn(y_pred, torch.reshape(ybatch, (-1,)).to(torch.int64))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                print(f'Split {split_index}, Epoch {epoch}, latest loss {loss}')
            torch.save(model.state_dict(), f"model/PPMImodels/Model_{split_index}")
        train_model(model, optimizer, X_train, y_train, loss_fn, n_epochs)
        eval_model(model, X_test, y_test)
# if a cli argument is given only evaluate
else: 
    if os.path.exists("model/PPMImodels/GlobalModel"):
        model = DiabModel()
        model.load_state_dict(torch.load("model/PPMImodels/GlobalModel"))
        X_test_all = torch.tensor(fullset[:, 2:], dtype=torch.float32)
        y_test_all = torch.tensor(fullset[:, 1], dtype=torch.float32).reshape(-1, 1)
        eval_model(model, X_test_all, y_test_all)
    else: 
        print("no global model found")
        quit()
        
        
