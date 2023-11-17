#include <iostream>
#include <fstream>
#include<vector>
#include<string>
#include<tuple>
#include <boost/format.hpp>
#include "Constants.h"


using namespace std;
using boost::format;

bool is_file_existing(const string &file_name) {
    ifstream infile(file_name);
    auto result = infile.good();
    return result;
}


/*
* Reads the first delta file to determine the length of delta
* h and delta vectors are later used to write all corresponding values into them
* returns the length of delta for a single model
*/
uint8_t find_delta_size(string directory, ROLE_TYPE role){
    string file_name = str(format("%s%s_C000.txt") % directory % role);
    cout << "Read size from: " << file_name << endl;
    std::fstream input_file(file_name, std::ios_base::in);
    vector<NUMBER_TYPE> counter;

    SIGNED_NUMBER_TYPE a;
    while (input_file >> a) {
        cout << a << endl;
        counter.push_back(a);
    }
    cout << "size = " << counter.size() -1 << endl;
    return counter.size() -1;  // 
}

/*
* Returns 2 arrays. One for the delta's of a single model and one for the h value of the model
*/
tuple<NUMBER_TYPE *, NUMBER_TYPE *> read_deltas_and_h(const string &file_name, uint32_t number_of_entries) {
    cout << number_of_entries << endl;
    auto *delta_values = new NUMBER_TYPE[number_of_entries];
    auto *h_value = new NUMBER_TYPE[1];
    uint32_t next_entry_to_read = 0;

    std::fstream input_file(file_name, std::ios_base::in);
    SIGNED_NUMBER_TYPE a;
    while (input_file >> a) {
        if(next_entry_to_read == 0){
            h_value[0] = a; 
            cout << "filling h " << next_entry_to_read << endl;
        }
        else {
            cout << "filling delta array " << next_entry_to_read-1 << endl;
            delta_values[next_entry_to_read-1] = a;
        }
        assert(next_entry_to_read < number_of_entries+1);  // +1 because next_entry_to_read will read including the h value so it will 1 larger than our delta size
        next_entry_to_read++;
    }
    assert(next_entry_to_read == number_of_entries+1);
    return {delta_values, h_value};
}


/*
* Reads all models and creates 2 vectors containing 
*/
tuple<uint32_t, tuple<vector<NUMBER_TYPE *> *, vector<NUMBER_TYPE *> *>> read_q_fed_avg_data(string directory, ROLE_TYPE role,
                                                           size_t max_models_to_read){
    auto *delta_values  = new vector<NUMBER_TYPE*>();
    auto *h_vec = new vector<NUMBER_TYPE*>();
    //auto *client_models = new vector<NUMBER_TYPE *>();
    auto number_of_entries_in_delta = find_delta_size(directory, role);

    uint32_t current_client_index = 0;
    string file_name = str(format("%s%s_C%03u.txt") % directory % role % current_client_index);

    while ((max_models_to_read == 0 || delta_values->size() < max_models_to_read) && is_file_existing(file_name)) {

        cout << "Read local model from: " << file_name << endl;
        auto deltas_and_h = read_deltas_and_h(file_name, number_of_entries_in_delta);
        cout << "reading successful" << file_name << endl;
        NUMBER_TYPE* curr_deltas  = get<0>(deltas_and_h);
        NUMBER_TYPE* curr_h = get<1>(deltas_and_h);
        cout << "after access tuples" << file_name << endl;
        delta_values->push_back(curr_deltas);
        h_vec->push_back(curr_h);
        cout << "after vector pushes " << file_name << "size delta: "<<delta_values->size() << " and size h: " <<h_vec->size() << endl;
        current_client_index++ ;
        cout << "++ " << endl;
        //file_name = str(format("%s%s_C%03u.txt") % directory % role % current_client_index);
        cout << file_name << endl;
        cout << "endloop " << endl;
    }
    return {number_of_entries_in_delta, {delta_values, h_vec}};

}

tuple<uint32_t, NUMBER_TYPE *> read_first_model(string directory, ROLE_TYPE role) {

    string file_name = str(format("%s%s_C000.txt") % directory % role);
    cout << "Read local model from: " << file_name << endl;
    std::fstream input_file(file_name, std::ios_base::in);
    vector<NUMBER_TYPE> counter;

    SIGNED_NUMBER_TYPE a;
    while (input_file >> a) {
        counter.push_back(a);
    }
    auto *model_as_array = new SIGNED_NUMBER_TYPE[counter.size()];
    copy(counter.begin(), counter.end(), model_as_array);
    return {counter.size(), (NUMBER_TYPE *) model_as_array};
}

NUMBER_TYPE *read_model(const string &file_name, uint32_t number_of_entries) {

    auto *result = new SIGNED_NUMBER_TYPE[number_of_entries];
    uint32_t next_entry_to_read = 0;

    std::fstream input_file(file_name, std::ios_base::in);
    SIGNED_NUMBER_TYPE a;
    while (input_file >> a) {
        assert(next_entry_to_read < number_of_entries);
        result[next_entry_to_read] = a;
        next_entry_to_read++;
    }
    assert(next_entry_to_read == number_of_entries);

    return (NUMBER_TYPE *) result;
}

NUMBER_TYPE *_read_weights(const string &file_name, uint32_t number_of_entries) {

    auto *result = new SIGNED_NUMBER_TYPE[number_of_entries];
    uint32_t next_entry_to_read = 0;

    std::fstream input_file(file_name, std::ios_base::in);
    SIGNED_NUMBER_TYPE a;
    while (input_file >> a) {
        assert(next_entry_to_read < number_of_entries);
        result[next_entry_to_read] = a;
        next_entry_to_read++;
    }
    assert(next_entry_to_read == number_of_entries);

    return (NUMBER_TYPE *) result;
}


tuple<uint32_t, vector<NUMBER_TYPE *> *> read_local_models(string directory, ROLE_TYPE role,
                                                           size_t max_models_to_read) {
    auto *client_models = new vector<NUMBER_TYPE *>();
    auto number_of_entries_and_first_model = read_first_model(directory, role);
    uint32_t number_of_entries_per_model = get<0>(number_of_entries_and_first_model);
    client_models->push_back(get<1>(number_of_entries_and_first_model));

    unsigned int current_client_index = 1;
    string file_name = str(format("%s%s_C%03u.txt") % directory % role % current_client_index);
    while ((max_models_to_read == 0 || client_models->size() < max_models_to_read) && is_file_existing(file_name)) {

        cout << "Read local model from: " << file_name << endl;
        client_models->push_back(read_model(file_name, number_of_entries_per_model));
        current_client_index += 1;
        file_name = str(format("%s%s_C%03u.txt") % directory % role % current_client_index);
    }

    cout << "Read " << client_models->size() << " models with " << number_of_entries_per_model << " entries each from "
         << directory
         << endl;

    return {number_of_entries_per_model, client_models};
}

NUMBER_TYPE *read_global_model(string directory, uint32_t number_of_entries) {

    string file_name = str(format("%sglobal.txt") % directory);
    cout << "Read global model from: " << file_name << endl;
    return read_model(file_name, number_of_entries);

}

NUMBER_TYPE *read_weights(string directory, uint32_t number_of_entries) {

    string file_name = str(format("%sq_vals.txt") % directory);
    
    if (is_file_existing(file_name)){
        cout << "Read Q-values from: " << file_name << endl;
        return _read_weights(file_name, number_of_entries);
    }
    else {
        cout << "create Q-values all : "<< number_of_entries << " " << file_name << endl;
        ofstream new_file(file_name.c_str());
        if (new_file.is_open()){

            for (uint32_t i = 0; i < number_of_entries; i++){
                new_file << number_of_entries << endl;
            }
            new_file.close();
            return _read_weights(file_name, number_of_entries);
        }
        else {
            cout << "Failed to create: " << file_name << endl;
            return nullptr;
        }
    }
}


void save_model(uint32_t number_of_entries_per_model, const string &file_name, OUTPUT_NUMBER_TYPE *model) {
    auto *signed_model = (SIGNED_OUTPUT_NUMBER_TYPE *) model;

    std::ofstream ofile;
    ofile.open(file_name, std::ios_base::out);

    for (uint32_t i = 0; i < number_of_entries_per_model; i++) {
        if (i > 0) {
            ofile << std::endl;
        }
        ofile << signed_model[i];
    }
    ofile << std::endl;
    ofile.close();
}

void send_aggregated_model(string directory, uint32_t number_of_entries_per_model, OUTPUT_NUMBER_TYPE *model,
                           ROLE_TYPE role) {
    auto file_name_model = str(format("%sAggregatedModel_%s.txt") % directory % role);
    save_model(number_of_entries_per_model, file_name_model, model);
}
