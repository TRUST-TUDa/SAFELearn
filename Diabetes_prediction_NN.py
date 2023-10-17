import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def compute_recall(TP, FN):
    return TP/ (TP + FN + 1e-8)

def compute_precision(TP, FP):
    return TP / (TP + FP + 1e-8)

def compute_f1_score(precision, recall):
    # Convert predictions to boolean values (0 or 1)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
    return f1_score.item()


# load the dataset, split into input (X) and output (y) variables
dataset = np.loadtxt('data/Prepped_diabetes_data.data', delimiter=',')
TestSet = np.loadtxt('data/Prepped_diabetes_data.data', delimiter=',')
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
    TP = (y_test * y_pred).sum()
    FP = ((1 - y_test) * y_pred).sum()
    FN = (y_test * (1 - y_pred)).sum()
    
    recall = compute_recall(TP, FN)
    precision = compute_precision(TP,FP)
    f1_score = compute_f1_score(precision, recall)
    print(f"Accuracy {accuracy}")
    print(f"F1 Score: {f1_score}")
    print(f"Recall: {recall}")
    print(f"Precision: {precision}")

torch.save(model, "model/myModel")

