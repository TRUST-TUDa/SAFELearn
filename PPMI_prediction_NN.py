import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import sys
import sklearn.metrics as sklm
import os
import glob
from torcheval.metrics.functional import multiclass_f1_score, multiclass_auroc

# Change constants here
###############################################################################
LIPSCHITZCONSTANT = 1
Q_FACTOR = 0
TORCHSEED = 42
DEFAULT_DEVICE = "cpu"
NUMBER_OF_CLIENTS =3
PROJECT = "PPMI"
INPUT_DATA_PATH = f"input_data/{PROJECT}/PPMI_cleaned_altered.csv"
MODEL_PATH= f"model/{PROJECT}/"
GLOBAL_MODEL_PATH = f"{MODEL_PATH}/GlobalModel.txt"
###############################################################################

# INIT
###############################################################################
torch.manual_seed(TORCHSEED)
generator = torch.Generator().manual_seed(TORCHSEED)

device = torch.device(DEFAULT_DEVICE)
print("Device:", device)

if not os.path.exists("model"):
        os.mkdir("model")

if not os.path.exists(MODEL_PATH):
        os.mkdir(MODEL_PATH)
###############################################################################


fullset = pd.read_csv(INPUT_DATA_PATH)
fullset = torch.Tensor(fullset.to_numpy())

set_size = len(fullset)
clients = []

# Split the data into non-overlapping parts
split_size = len(fullset) // NUMBER_OF_CLIENTS 
for client_index in range(NUMBER_OF_CLIENTS):
    split = fullset[client_index * split_size: (client_index + 1) * split_size]
    clients.append(split)

n_epochs = 50
batch_size = 64

class PPMIModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(12,20)
        self.linear2 = nn.Linear(20,15)
        self.linear3 = nn.Linear(15,12)
        self.linear4 = nn.Linear(12,4)
        self.linear5 = nn.Linear(4,3)
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
        return x

def eval_model(model, X_test, y_test):
    model.eval()

    with torch.no_grad():
        y_pred = model(X_test)
        y_pred_integer = y_pred.round().cpu().numpy()
        print(multiclass_f1_score(y_pred, torch.reshape( y_test, (-1, )), num_classes=3))
        print(multiclass_auroc(y_pred, torch.reshape( y_test, (-1, )), num_classes=3))


# if the global model does not yet exist create a new fully untrained one
if not os.path.exists(GLOBAL_MODEL_PATH):
    model = PPMIModel()
    torch.save(model.state_dict(), GLOBAL_MODEL_PATH)

global_model = PPMIModel()
global_model.load_state_dict(torch.load(GLOBAL_MODEL_PATH))

def delete_files(file_pattern):
    files_to_delete = glob.glob(file_pattern)
    for file in files_to_delete:
        os.remove(file)

# delete all existing client models, losses, and delta files
delete_files(f"{MODEL_PATH}Model_*")
delete_files(f"{MODEL_PATH}Loss_*")
delete_files(f"{MODEL_PATH}Delta_*")




for client_index, split_data in enumerate(clients):
    train_size = int(0.8 * len(split_data))
    test_size = len(split_data) - train_size
    train_data, test_data = data.random_split(split_data, [train_size, test_size], generator=generator)
        
    X_train = torch.tensor(train_data.dataset[:, 2:], dtype=torch.float32)
    y_train = torch.tensor(train_data.dataset[:, 1], dtype=torch.float32).reshape(-1, 1)
    X_test = torch.tensor(test_data.dataset[:, 2:], dtype=torch.float32)
    y_test = torch.tensor(test_data.dataset[:, 1], dtype=torch.float32).reshape(-1, 1)
    model = PPMIModel()
        
    # if there exists a global model from earlier learnings import it
    model.load_state_dict(torch.load(GLOBAL_MODEL_PATH))
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
            print(f'Client {client_index}, Epoch {epoch}, latest loss {loss}')
        torch.save(model.state_dict(), f"{MODEL_PATH}Model_{client_index}.txt")
        #torch.save(loss_fn(y_pred, torch.reshape(y_train, (-1,)).to(torch.int64)), f"model/PPMImodels/Loss_{client_index}"
    ## execute the code    

    if (len(sys.argv) == 1):
        train_model(model, optimizer, X_train, y_train, loss_fn, n_epochs)
    eval_model(model, X_test, y_test)
        


def get_one_vec_sorted_layers(model):
    layer_names = model.keys()
    size = 0
    for name in layer_names:
        size += model[name].view(-1).shape[0]
    sum_var = torch.FloatTensor(size).fill_(0)
    size = 0
    for name in layer_names:
        layer_as_vector = model[name].view(-1)
        layer_width = layer_as_vector.shape[0]
        sum_var[size:size + layer_width] = layer_as_vector
        size += layer_width
    return sum_var



def calculate_delta_wt(global_model, model, L):
    vec_glob = get_one_vec_sorted_layers(global_model.state_dict())
    vec_mod = get_one_vec_sorted_layers(model.state_dict())
    return L * (vec_glob - vec_mod)

def calculate_delta(q, loss, deltawt):
    return loss ** q * deltawt

def calculate_ht(q, loss, deltawt, L):
    return q * loss.detach().numpy() ** q * np.linalg.norm(deltawt.detach().numpy(),2) + loss.detach().numpy() ** q * L

y_pred = global_model(X_train)
loss = loss_fn(y_pred, torch.reshape(y_train, (-1,)).to(torch.int64))

for client_index in range(len(clients)):
    model = PPMIModel()
    model.load_state_dict(torch.load(f"{MODEL_PATH}Model_{client_index}.txt"))
    deltawt = calculate_delta_wt(global_model, model, LIPSCHITZCONSTANT)
    delta = calculate_delta(Q_FACTOR, loss, deltawt)
    ht = calculate_ht(Q_FACTOR, loss, deltawt, LIPSCHITZCONSTANT)
    combined = np.concatenate((np.array([ht]), delta.detach().numpy()))
    np.savetxt(f"{MODEL_PATH}Delta_{client_index}.txt", combined, fmt='%.8f')
    #f.write(delta.numpy() + "\n" + ht.numpy())
    #f.close()