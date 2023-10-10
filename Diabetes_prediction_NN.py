import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
 
# load the dataset, split into input (X) and output (y) variables
dataset = np.loadtxt('Prepped_diabetes_data.data', delimiter=',')
TestSet = np.loadtxt('Prepped_diabetes_data.data', delimiter=',')
X = dataset[:, :8]
y = dataset[:,8]
 
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
 
# define the model
model = nn.Sequential(
    nn.Linear(8, 16),
    nn.ReLU(),
    nn.Linear(16, 10),
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
optimizer = optim.Adam(model.parameters(), lr=0.01)
 
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

X_test = TestSet[:, :8]
y_test = TestSet[:,8]
X_test = torch.tensor(X, dtype=torch.float32)
y_test = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

# compute accuracy (no_grad is optional)
with torch.no_grad():
    y_pred = model(X_test)
accuracy = (y_pred.round() == y_test).float().mean()
print(f"Accuracy {accuracy}")

torch.save(model, "myModel")
