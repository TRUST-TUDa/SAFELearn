# Process to train and aggregate local model with global model

## Train local model
To Train the local model use the desired Modelfile e.g.
```python3 Diabetes_prediction_NN.py local```
This will generate a local model and save it as model/Localmodel

## Create splits
Use the file Split_Aggregate.py like this
```python3 Split_Aggregate.py split```
This will create splits in the Directory

## Use ABY to aggregate local model of the client with global model of the server

for the server this is: 
```./build/fedavg_aggregation -n 100 -r 0 -d "MyTestDir"```

for the client this is: 
```./build/fedavg_aggregation -n 100 -r 1 -d "MyTestDir"```

this will create 2 outputs which can be aggregated together. It is saved in data/Aggregated as AggregatedModel_A.txt and AggregatedModel_B.txt

## Combining the 2 Aggregated_Model 
```python3 Split_Aggregate.py aggregate```
This will save a newly combined model in model/NewModel

## Use the new model
```python3 UseNewDiabModel.py```
