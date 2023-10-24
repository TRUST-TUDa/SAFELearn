import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
# use seaborn plotting defaults
#import seaborn as sns; sns.set()
from sklearn.svm import LinearSVC, SVC
import torch
import torch.utils.data as loader
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import sys as sys
from joblib import dump, load



if(len(sys.argv)==1):
    print("For GlobalModel use: python3", sys.argv[0], "global\nFor LocalModel use: python3", sys.argv[0], "local")
    quit()

version = sys.argv[1]



# load the dataset, split into input (X) and output (y) variables
fullset = np.loadtxt('data/Prepped_diabetes_data.data', delimiter=',')
#TestSet = np.loadtxt('data/Prepped_diabetes_data.data', delimiter=',')

generator1 = torch.Generator().manual_seed(42)
generator2 = torch.Generator().manual_seed(42)

train_size = int(0.8 * len(fullset))
test_size = len(fullset) - train_size
trainset, TestSet = loader.random_split(fullset, [train_size, test_size], generator=generator1)
    
dataLoader = loader.DataLoader(fullset)


X_train = trainset.dataset[:, :8]
y_train = trainset.dataset[:,8]
 
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)

X_test = TestSet.dataset[:, :8]
y_test = TestSet.dataset[:,8]
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.int32).reshape(-1, 1)



# defining parameter range 
param_grid = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}  
  
#model = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3) 


#best params we found
model = SVC(C=1000, gamma=0.0001, kernel='rbf')

print("Model pramaters", model.get_params())

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(classification_report(y_true=y_test, y_pred=y_pred, digits=3))



dump(model, 'SVM_trained-model.joblib') 



print(model.get_params())