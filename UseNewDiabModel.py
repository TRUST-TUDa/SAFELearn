import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as loader
import sys as sys

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

train_size = int(0.8 * len(fullset))
test_size = len(fullset) - train_size
trainset, TestSet = loader.random_split(fullset, [train_size, test_size], generator=generator1)

dataLoader = loader.DataLoader(fullset)


X = trainset.dataset[:][:, :8]
y = trainset.dataset[:][:,8]
 
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
 
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
model.load_state_dict(torch.load("model/NewModel"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

X_test = TestSet.dataset[:][:, :8]
y_test = TestSet.dataset[:][:,8]
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

# compute accuracy (no_grad is optional)
with torch.no_grad():
    y_pred = model(X_test)
    y_pred_binary = y_pred.round()
    print(y_test)
    accuracy = (y_pred_binary == y_test).float().mean()
    TP = ((y_pred_binary == 1) & (y_test == 1)).float().sum()
    FP = ((y_pred_binary == 1) & (y_test == 0)).float().sum()
    FN = ((y_pred_binary == 0) & (y_test == 1)).float().sum()
    
    recall = compute_recall(TP, FN)
    precision = compute_precision(TP,FP)
    f1_score = compute_f1_score(precision, recall)
    print(f"Accuracy {accuracy}")
    print(f"F1 Score: {f1_score}")
    print(f"Recall: {recall}")
    print(f"Precision: {precision}")

