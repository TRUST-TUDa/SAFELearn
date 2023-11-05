import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import sys as sys
import sklearn.metrics as sklm


torch.manual_seed(42)


if(len(sys.argv)==1):
    print("For GlobalModel use: python3", sys.argv[0], "global\nFor LocalModel use: python3", sys.argv[0], "local\nFor evaluation use: python3", sys.argv[0], "eval")
    quit()

version = sys.argv[1]

# determine device (for gpu support)
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
#if torch.backends.mps.is_available(): # currently the mps device seems to be much slower than the cpu so its commented out
#    device = torch.device("cpu")
#    torch.mps.manual_seed(42)

print("Device:", device)
fullset = np.loadtxt('data/Prepped_diabetes_data.data', delimiter=',')

generator1 = torch.Generator().manual_seed(42)
generator2 = torch.Generator().manual_seed(42)
torch.manual_seed(42)

set_size = int(0.5 * len(fullset))
SetGlobal, SetLocal = data.random_split(fullset, [set_size, set_size], generator=generator1)

train_size = int(0.8 * len(SetGlobal))
test_size = len(SetGlobal) - train_size

if(version == "global" or version == "eval"):
    trainset, TestSet = data.random_split(SetGlobal, [train_size, test_size], generator=generator1)
    dataLoader = data.DataLoader(SetGlobal)
if(version == "local"):
    trainset, TestSet = data.random_split(SetLocal, [train_size, test_size], generator=generator2)
    dataLoader = data.DataLoader(SetLocal)


X_train = trainset.dataset[:][:, :8]
y_train = trainset.dataset[:][:,8]
 
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)

X_test = TestSet.dataset[:][:, :8]
y_test = TestSet.dataset[:][:,8]
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

class DiabModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(8,20)
        self.linear2 = nn.Linear(20,10)
        self.linear3 = nn.Linear(10,4)
        self.linear4 = nn.Linear(4,1)
        self.act_fn = nn.ReLU()
        self.sigm = nn.Sigmoid()


    def forward(self, x):
        x = self.linear1(x)
        x = self.act_fn(x)
        x = self.linear2(x)
        x = self.act_fn(x)
        x = self.linear3(x)
        x = self.act_fn(x)
        x = self.linear4(x)
        x = self.sigm(x)
        return x

model = DiabModel()
if (sys.argv[1]== "eval"):
    model.load_state_dict(torch.load("model/NewModel"))
else: 
    print(model)

loss_fn   = nn.BCELoss()  # binary cross entropy with a sigmoid end layer (numerically more stable)  
optimizer = optim.Adam(model.parameters(), lr=0.001) 
 
n_epochs = 100
batch_size = 64



def train_model(model,optimizer, X_train, y_train, loss_fn, n_epochs=100):
    model.train()
    model.to(device)
    for epoch in range(n_epochs):
        for i in range(0, len(X_train), batch_size): # make this random
            Xbatch = X_train[i:i+batch_size].to(device)
            ybatch = y_train[i:i+batch_size].to(device)
            y_pred = model(Xbatch)
            loss = loss_fn(y_pred, ybatch)
            optimizer.zero_grad() #Zero gradiants (necessary for backpropagation) - otherwise they are accumulated over time
            loss.backward() # Backpropagation
            optimizer.step() # Update parameters
        print(f'Finished epoch {epoch}, latest loss {loss}')
if (not sys.argv[1]=="eval"):
    train_model(model,optimizer,X_train,y_train,loss_fn, n_epochs)

def eval_model(model, X_test, y_test):
    model.eval()

    with torch.no_grad():
        y_pred = model(X_test)
        y_pred_binary = y_pred.round().cpu().numpy()
        precision_sklm = sklm.precision_score(y_test, y_pred_binary)
        recall_sklm = sklm.recall_score(y_test, y_pred_binary)
        f1score_sklm = sklm.f1_score(y_test, y_pred_binary)
        accuracy_sklm = sklm.accuracy_score(y_test, y_pred_binary)
    
        print("prec: ", precision_sklm)
        print("recall: ", recall_sklm)
        print("f1score: ", f1score_sklm)
        print("accuracy ", accuracy_sklm)

eval_model(model,X_test, y_test)

if(version =="global"):
    torch.save(model.state_dict(), "model/GlobalModel")
if(version =="local"):
    torch.save(model.state_dict(), "model/LocalModel")

