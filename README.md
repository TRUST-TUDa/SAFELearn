# Overview
This code provides the implementation of SAFELearn, a scheme for Secure Aggregation for private FEderated Learning. The code allows to securely aggregating model updates without revealing them to the aggregator using secure multi party computation.
The corresponding paper was published as Fereidooni, Hossein, et al. "SAFELearn: Secure Aggregation for private FEderated Learning." 2021 IEEE Security and Privacy Workshops (SPW). IEEE, 2021. [Paper available here.](https://encrypto.de/papers/FMMMMNRSSYZ21.pdf)

```bibtex
@inproceedings{fereidooni2021safelearn,
  title={SAFELearn: Secure Aggregation for private FEderated Learning},
  author={Fereidooni, Hossein and Marchal, Samuel and Miettinen, Markus and Mirhoseini, Azalia and M{\"o}llering, Helen and Nguyen, Thien Duc and Rieger, Phillip and Sadeghi, Ahmad-Reza and Schneider, Thomas and Yalame, Hossein and others},
  booktitle={2021 IEEE Security and Privacy Workshops (SPW)},
  pages={56--62},
  year={2021},
  organization={IEEE}
}
```

This code is provided as a experimental implementation for testing purposes and should not be used in a productive environment. We cannot guarantee security or correctness.

The code need to be linked against a shared library with the implementation of ABY. An implementation is available [here](https://github.com/encryptogroup/ABY) but also every other implementation of the ABY header files can be linked.

# Setup
1. Install the required librariries: gmpxx gmp pthread boost_system crypto dl backtrace
1. Download ABY, compile it and change the path to its install location in CMAKELists.txt (ABY_PATH)
2. create the build directory, cd into it and call:
```
cmake ..
```
3. Build the project by running
```
make
```

# Setup with Docker
1. Install docker
2. Get the image: 
```
docker pull maettu102/safelearn:latest &&
```
3. Run the container with the image (name it safelearn):
```
docker run -itd --name safelearn maettu102/safelearn:latest &&
```
4. Connect to the container
```
docker exec -it safelearn /bin/bash
```
3. You should get a bash shell in the Workdir /Safelearn
4. The executable is in the build directory

# Running the code
Some example test files are provided in the directory data, all constants in the code are already adapted to run this. If different data should be used, the constant DATA_DIR in constants.h need to be adapted. Furthermore, the IP address of the other server need to be changed. To ease the execution, currently 127.0.0.1 is set (although the experiments in the SAFELearn paper were executed on different servers). It should be noted that using localhost instead of an IP address does not seem to work.
The test scenario can be executed by running 
```
./fedavg_aggregation -r 0 -n 100 -d "TestExample"
```
on the server and 
```
./fedavg_aggregation -r 1 -n 100 -d "TestExample"
```
on the client.

# Code Structure
## Splitting Model Update
Each client first needs to split its model update u into two vectors a and b, s.t. u=a+b. After the aggregation is finished, the SMC part produces two vectors g1 and g2, s.t. the aggregated model a can be obtained as g1+g2.
The file SplitGenerator.py provides the necesarry functions to do both conversions for pytorch state dicts. It also takes care of the conversion from floating point numbers to fixed point numbers.
### Preprocessing
The function create_splits takes the file paths to the local models and the global models, the target directory (which must not exist before running the script) and a sort list of all layer names in the state dict. It loads the models, performs the conversions and writes the splits to the provided target directory.
The purpose of the layer names is to ensure a specific order of the parameters, when concatenating all parameters of a single model to obtain a representation as vector. Therefore, this list must be the same when determining the aggregated model from the output of the MPC part.
### Post Processing
The function determine_aggregated_model takes the paths to both shares of the aggregated model, the list of layer names and an example model. It returns the aggregated model as dictionary.
The example model is used for determining the dimensions of the individual layers.
##  Secure Multi Party Computation
### Entry Point
The SMC part follows the structure of the examples of ABY. The main function is in the file FedAvgModelAggregationTest.cpp, which loads the models and provides them to the smc part by calling aggregate_models from MPCAggregator.h.
### Aggregation
The actual SMC happens in the directory mpcaggregator. Here, the code in MPCAggregator.cpp initiates ABY and converts the given input arrays into ABY shares. The aggregation then happens in the file Aggregator.cpp
### MPC Utils
The files in the directory mpcutils provide technical helper classes to simplify the usage of ABY. In particular, here wrappers for the circuits and shares of ABY are defined, to ensure compile errors when an arithmetic share is used as a Yao share. For this wrapper classes for all share types exist in the file MPCShare.h . The circuit classes in MPCCircuit.h hold the actual circuit from ABY and simply unwrap the actual share, performing some safety checks (e.g., to ensure that SIMD shares have the same length), forward the call to the actual circuit and return a wrapped version of the resulting ABY share.
### Utils
The files in the directory utils provides constants/typedefs and utility functions that are not related to ABY. In particular, functions for loading the shares from text files (cf. read_local_models in ClientServerConnector.cpp), writing shares to text files (cf. send_aggregated_model in ClientServerConnector.cpp)

