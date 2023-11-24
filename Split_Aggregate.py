import torch
import os
import numpy as np
import sys
import glob

# Change constants here
###############################################################################
PUSH_FACTOR = 2 ** 10
LIMIT = (2 ** 3) * PUSH_FACTOR

PROJECT = "PPMI"
# path where the new global model will be saved after combining the splits
NEW_MODEL_PATH = f"model/{PROJECT}/GlobalModel.txt"
GLOBAL_MODEL_PATH = f"model/{PROJECT}/GlobalModel.txt" # normally the same as NEW_MODEL_PATH
SPLITTED_FILE_DIR = f"data/{PROJECT}Splits"
MAX_MODELS = 10000
TORCHSEED = 42
###############################################################################

# INIT
###############################################################################
# mode of this document c = combine, s = split, q = q-split
print("which mode should be selected? c = combine, s = split, q = q-split")
mode = input()

if not os.path.exists(SPLITTED_FILE_DIR):
        os.mkdir(SPLITTED_FILE_DIR)


torch.manual_seed(TORCHSEED)
###############################################################################

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


def restrict_values(vec):
    vec = PUSH_FACTOR * vec
    vec = torch.round(vec)
    vec = torch.clamp(vec, -LIMIT, LIMIT)
    restricted_vec = vec.type(torch.LongTensor)
    return restricted_vec


def unrestrict_values(recovered_restricted_vec):
    recovered_restricted_vec = recovered_restricted_vec.type(torch.FloatTensor)
    return recovered_restricted_vec / PUSH_FACTOR


def split(restricted_vec):
    a = torch.LongTensor(restricted_vec.shape).random_(-LIMIT, LIMIT)
    b = restricted_vec - a
    safety_counter = 0
    while True:
        indices_to_recompute = torch.nonzero(torch.abs(b) >= LIMIT)
        if len(indices_to_recompute) == 0:
            break
        if safety_counter > 100:
            raise Exception('Did not find suitable randomvalues')
        indices_to_recompute = indices_to_recompute.view(-1)
        #print(f'\tRegenerate {indices_to_recompute.shape[0]} elements (from {restricted_vec.shape[0]})')
        a[indices_to_recompute] = torch.LongTensor(restricted_vec[indices_to_recompute].shape).random_(-LIMIT, LIMIT)
        b = restricted_vec - a
        safety_counter += 1
    return a, b


def delete_files(file_pattern):
    files_to_delete = glob.glob(file_pattern)
    for file in files_to_delete:
        os.remove(file)


def split_global_model(global_model_path):
    delete_files(f"{SPLITTED_FILE_DIR}/A*")
    delete_files(f"{SPLITTED_FILE_DIR}/B*")
    global_model = torch.load(global_model_path)
    global_model_as_vec = get_one_vec_sorted_layers(global_model)
    restricted_vec = restrict_values(global_model_as_vec)    
    np.savetxt(f'{SPLITTED_FILE_DIR}/global.txt', restricted_vec.numpy(), fmt='%d')


def create_splits(global_model_path, local_model_paths, q=False):
    split_global_model(global_model_path)
    for i, path in enumerate(local_model_paths):
        vec = ""
        if q:
            delta_wk_h_np = np.loadtxt(path)
            vec = torch.tensor(delta_wk_h_np)
            delta_part = restrict_values(vec[1:])
            restricted_local_vec = torch.tensor(np.concatenate((vec[0].numpy().reshape(1,),delta_part.numpy())))
        else:
            local_model = torch.load(path)
            vec = get_one_vec_sorted_layers(local_model)
            restricted_local_vec = restrict_values(vec) 

           
        print("restrticted ", restricted_local_vec)
        a, b = split(restricted_local_vec)
        a_file = f'{SPLITTED_FILE_DIR}/A_C{i:03d}.txt'
        b_file = f'{SPLITTED_FILE_DIR}/B_C{i:03d}.txt'
        np.savetxt(a_file, a.numpy(), fmt='%d')
        np.savetxt(b_file, b.numpy(), fmt='%d')
    print(f"Splitted {len(local_model_paths)} models into {SPLITTED_FILE_DIR}")


def create_q_splits(global_model_path, delta_wk_h_paths):
    create_splits(global_model_path,delta_wk_h_paths, q=True)


def determine_number_of_entries_in_matrix(shape):
    result = 1
    for dimension in shape:
        result *= dimension
    return result


def recover_model_from_vec(example_model, vec_to_recover, layer_names):
    result = {}
    start_index_of_next_layer = 0
    for layer_name in layer_names:
        expected_shape = example_model[layer_name].shape
        entries_in_layer = determine_number_of_entries_in_matrix(expected_shape)
        end_index_of_current_layer = start_index_of_next_layer + entries_in_layer
        entries = vec_to_recover[start_index_of_next_layer: end_index_of_current_layer]
        result[layer_name] = entries.view(expected_shape)
        start_index_of_next_layer += entries_in_layer
    return result


def determine_aggregated_model(old_global_model_path, path_to_share1, path_to_share2):
    old_global_model = torch.load(old_global_model_path)
    layer_names = old_global_model.keys()
    share1 = np.loadtxt(path_to_share1, dtype=np.int64)
    share2 = np.loadtxt(path_to_share2, dtype=np.int64)
    restricted_vec = share1 + share2
    np.savetxt("./data/Aggregated/combined", restricted_vec, fmt='%d') #used for debugging purposes 
    unrestricted_vec = unrestrict_values(torch.from_numpy(restricted_vec))
    return recover_model_from_vec(old_global_model, unrestricted_vec, layer_names)

def get_models_as_list(filename_without_i):
    localmodelpaths = []
    for i in range(MAX_MODELS):
        filename = f"{filename_without_i}{i}.txt"
        if os.path.exists(filename):
            localmodelpaths.append(filename)
        else: 
            break
    return localmodelpaths



# Execution mode s = split
###############################################################################
if (mode == "s"):
    print(f"Splitting to data/{PROJECT}Splits")
    localmodelpaths = get_models_as_list(f"model/{PROJECT}/Model_")
    create_splits(GLOBAL_MODEL_PATH,localmodelpaths)

# Execution mode q = q-split
###############################################################################
if (mode == "q"):
    print(f"Splitting to data/{PROJECT}Splits")
    localmodelpaths = get_models_as_list(f"model/{PROJECT}/Delta_")
    create_q_splits(GLOBAL_MODEL_PATH,localmodelpaths)

# Execution mode c = combine
###############################################################################
if (mode == "c"):
    print("Aggregating! - new model will be saved at", NEW_MODEL_PATH)
    newmodel = determine_aggregated_model(GLOBAL_MODEL_PATH, "./data/Aggregated/AggregatedModel_A.txt", "./data/Aggregated/AggregatedModel_B.txt")
    torch.save(newmodel, NEW_MODEL_PATH)
