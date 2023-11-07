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

tuple<uint32_t, NUMBER_TYPE *> read_first_model(string directory, ROLE_TYPE role) {

    string file_name = str(format("%s%s_C000.txt") % directory % role);
    cout << "Read local model from: " << file_name << endl;
    std::fstream input_file(file_name, std::ios_base::in);
    vector<NUMBER_TYPE> model_of_client;

    SIGNED_NUMBER_TYPE a;
    while (input_file >> a) {
        model_of_client.push_back(a);
    }
    auto *model_as_array = new SIGNED_NUMBER_TYPE[model_of_client.size()];
    copy(model_of_client.begin(), model_of_client.end(), model_as_array);
    return {model_of_client.size(), (NUMBER_TYPE *) model_as_array};
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

NUMBER_TYPE *read_q(const string &file_name, uint32_t number_of_entries) {

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

NUMBER_TYPE *read_q_vals(string directory, uint32_t number_of_entries) {

    string file_name = str(format("%sq_vals.txt") % directory);
    
    if (is_file_existing(file_name)){
        cout << "Read Q-values from: " << file_name << endl;
        return read_q(file_name, number_of_entries);
    }
    else {
        cout << "create Q-values all 1: " << file_name << endl;
        ofstream new_file(file_name.c_str());
        if (new_file.is_open()){

            for (uint32_t i = 0; i < number_of_entries; i++){
                new_file << 1 << endl;
            }
            new_file.close();
            return read_q(file_name, number_of_entries);
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
    ofile.close();
}

void send_aggregated_model(string directory, uint32_t number_of_entries_per_model, OUTPUT_NUMBER_TYPE *model,
                           ROLE_TYPE role) {
    auto file_name_model = str(format("%sAggregatedModel_%s.txt") % directory % role);
    save_model(number_of_entries_per_model, file_name_model, model);
}
