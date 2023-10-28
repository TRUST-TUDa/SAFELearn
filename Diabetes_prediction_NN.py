import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as loader
import sys as sys
import sklearn as skl
import sklearn.metrics as sklm

if(len(sys.argv)==1):
    print("For GlobalModel use: python3", sys.argv[0], "global\nFor LocalModel use: python3", sys.argv[0], "local")
    quit()

version = sys.argv[1]

def compute_recall(TP, FN):
    return TP/ (TP + FN + 1e-8)

def compute_precision(TP, FP):
    return TP / (TP + FP + 1e-8)

def compute_f1_score(precision, recall):
    # Convert predictions to boolean values (0 or 1)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
    return f1_score.item()


# load the dataset, split into input (X) and output (y) variables
fullset = np.loadtxt('data/Prepped_diabetes_data.data', delimiter=',')
#TestSet = np.loadtxt('data/Prepped_diabetes_data.data', delimiter=',')

generator1 = torch.Generator().manual_seed(42)
generator2 = torch.Generator().manual_seed(42)
torch.manual_seed(42)


set_size = int(0.5 * len(fullset))
SetGlobal, SetLocal = loader.random_split(fullset, [set_size, set_size], generator=generator1)

train_size = int(0.8 * len(SetGlobal))
test_size = len(SetGlobal) - train_size

if(version == "global"):
    trainset, TestSet = loader.random_split(SetGlobal, [train_size, test_size], generator=generator1)
    dataLoader = loader.DataLoader(SetGlobal)
if(version == "local"):
    trainset, TestSet = loader.random_split(SetLocal, [train_size, test_size], generator=generator2)
    dataLoader = loader.DataLoader(SetLocal)



X = trainset.dataset[:][:, :8]
y = trainset.dataset[:][:,8]
 
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

X_test = TestSet.dataset[:][:, :8]
y_test = TestSet.dataset[:][:,8]
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.int32).reshape(-1, 1)
 
# define the model
model = nn.Sequential(
    nn.Linear(8, 20),
    nn.ReLU(),
    nn.Linear(20, 10),
    nn.ReLU(),
    #nn.Linear(12, 7),
    #nn.ReLU(),
    nn.Linear(10, 4),
    nn.ReLU(),
    #nn.Softmax()
    nn.Linear(4, 1),
    nn.Sigmoid()
)
print(model)

# train the model
loss_fn   = nn.BCELoss()  # binary cross entropy    # np.MSELoss
optimizer = optim.Adam(model.parameters(), lr=0.001) #try stuff
 
n_epochs = 100
batch_size = 100
 
for epoch in range(n_epochs):
    for i in range(0, len(X), batch_size): # make this random
        Xbatch = X[i:i+batch_size]
        
        y_pred = model(Xbatch)
        ybatch = y[i:i+batch_size]
        loss = loss_fn(y_pred, ybatch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Finished epoch {epoch}, latest loss {loss}')



# compute accuracy (no_grad is optional)
with torch.no_grad():
    X_test
    y_pred = model(X_test)
    y_pred_binary = y_pred.round()
    
    #print(y_test)
    #print(y_pred_binary)
    accuracy = (y_pred_binary == y_test).float().mean()
    TP = ((y_pred_binary == 1) & (y_test == 1)).float().sum()
    FP = ((y_pred_binary == 1) & (y_test == 0)).float().sum()
    FN = ((y_pred_binary == 0) & (y_test == 1)).float().sum()
    
    
    
    precision_sklm = sklm.precision_score(y_pred_binary, y_test)
    recall_sklm = sklm.recall_score(y_pred_binary, y_test)
    f1score_sklm = sklm.f1_score(y_pred_binary, y_test)
    accuracy_sklm = sklm.accuracy_score(y_pred_binary, y_test)
    
    print("prec: ", precision_sklm)
    print("recall: ", recall_sklm)
    print("f1score: ", f1score_sklm)
    print("accuracy ", accuracy_sklm)
    
    recall = compute_recall(TP, FN)
    precision = compute_precision(TP,FP)
    f1_score = compute_f1_score(precision, recall)
    print(f"Accuracy mättu: {accuracy}")
    print(f"F1 Score mättu: {f1_score}")
    print(f"Recall mättu: {recall}")
    print(f"Precision mättu: {precision}")

if(version =="global"):
    torch.save(model.state_dict(), "model/GlobalModel")
if(version =="local"):
    torch.save(model.state_dict(), "model/LocalModel")

