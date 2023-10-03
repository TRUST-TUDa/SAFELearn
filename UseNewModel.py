import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
 
# load the dataset, split into input (X) and output (y) variables


dataset = np.loadtxt('data/parkinsons_raw.data', delimiter=',')
TestSet = np.loadtxt('data/Parkinson_Test_Data.data', delimiter=',')

X = dataset[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,17,18,19,20,21,22]]
y = dataset[:,16]
print(X)
 
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
 
# define the model
model = nn.Sequential(
    nn.Linear(22, 30),
    nn.ReLU(),
    nn.Linear(30, 15),
    nn.ReLU(),
    nn.Linear(15, 10),
    nn.ReLU(),
    nn.Linear(10, 6),
    nn.ReLU(),
    nn.Linear(6, 1),
    nn.Sigmoid()
)

model.load_state_dict(torch.load("NewModel"))

X_test = TestSet[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,17,18,19,20,21,22]]
y_test = TestSet[:,16]
X_test = torch.tensor(X, dtype=torch.float32)
y_test = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

# compute accuracy (no_grad is optional)
with torch.no_grad():
    y_pred = model(X_test)
accuracy = (y_pred.round() == y_test).float().mean()
print(f"Accuracy {accuracy}")
torch.save(model.state_dict(),"./model/MyModel")
