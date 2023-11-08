#include "abycore/sharing/sharing.h"
#include "../utils/Constants.h"
#include "../mpcutils/MPCConstants.h"
#include <vector>
#include <tuple>
#include <memory>
#include <cstring>
#include <string>
#include "../mpcutils/MPCCircuit.h"
#include "../mpcutils/MPCParty.h"
#include "Aggregator.cpp"

using namespace std;

/// @brief determines the actual update -> subtracts the global from the SERVER split. 
/// @param global array of the numbers of the global model
/// @param local array of the numbers of one of the splits of the local model
/// @param result array where the result should be stored
/// @param number_of_entries the lenght of arrays of the models
/// @param add_zero true for the CLIENT, false for the SERVER
void determine_update(const NUMBER_TYPE *global, const NUMBER_TYPE *local, NUMBER_TYPE *result,
                      uint32_t number_of_entries, bool add_zero)
{
    for (uint32_t i = 0; i < number_of_entries; i++)
    {
        result[i] = local[i] - (add_zero ? 0U : global[i]);
    }
}

ArithmeticShare create_typed_update_share(uint32_t bitlen, e_role role, TypedArithmeticCircuit &circuit,
                                          NUMBER_TYPE *global_model, NUMBER_TYPE *local_model_share,
                                          uint32_t number_of_entries, NUMBER_TYPE *tmp)
{

    determine_update(global_model, local_model_share, tmp, number_of_entries, role == CLIENT);
    ArithmeticShare result = circuit->PutSharedSIMDINGate(number_of_entries, tmp, bitlen);
    return result;
}

NUMBER_TYPE *create_tmp_update_storage(uint32_t number_of_entries)
{
    return new NUMBER_TYPE[number_of_entries];
}

OUTPUT_NUMBER_TYPE *aggregate_models(e_role role, const string &address, uint16_t port, seclvl seclvl,
                                     e_mt_gen_alg mt_alg, NUMBER_TYPE *global_model,
                                     vector<NUMBER_TYPE *> *local_models,
                                     uint32_t number_of_entries, NUMBER_TYPE *q_vals)
{

    cout << get_time_as_string() << "Good Morning (bitlen=" << BIT_LENGTH << ")" << endl;
    MPCParty party = make_unique<MPCParty_>(role, address, port, seclvl, BIT_LENGTH, N_THREADS,
                                            mt_alg);

    TypedYaoCircuit yao_circuit = party->get_yao_circuit();
    TypedArithmeticCircuit arithmetic_circuit = party->get_arithmetic_circuit();

    seed_random_generator();

    vector<ArithmeticShare> clipped_updates;
    clipped_updates.reserve(local_models->size());

    NUMBER_TYPE *tmp_update_storage = create_tmp_update_storage(number_of_entries);
    for (auto &local_model : *local_models)
    {

        ArithmeticShare update = create_typed_update_share(BIT_LENGTH, role, arithmetic_circuit, global_model,
                                                           local_model, number_of_entries, tmp_update_storage);
        clipped_updates.push_back(update);
    }
    delete[] tmp_update_storage;

    auto *q_vals_vecs = new ArithmeticShare[number_of_entries];

    for (uint32_t i = 0; i < local_models->size(); i++){
        q_vals_vecs[i] = arithmetic_circuit->PutSIMDCONSGate(number_of_entries, q_vals[i], BIT_LENGTH);
    }

    ArithmeticShare global_model_share = arithmetic_circuit->PutSIMDINGate(number_of_entries, global_model, BIT_LENGTH, SERVER);

    OUTPUT_NUMBER_TYPE *aggregated_update = aggregate_models(party, BIT_LENGTH, number_of_entries, &clipped_updates,
                                                             arithmetic_circuit, yao_circuit, global_model_share, q_vals_vecs);

    auto plain_text_aggregated_update_shares = new vector<tuple<int32_t, OUTPUT_NUMBER_TYPE *>>();
    vector<size_t> client_numbers;
    client_numbers.reserve(plain_text_aggregated_update_shares->size());
    return aggregated_update;
}
