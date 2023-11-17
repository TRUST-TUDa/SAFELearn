#ifndef __MPC_AGGREGATOR_H_
#define __MPC_AGGREGATOR_H_

#include "../utils/Constants.h"
#include <vector>
#include <tuple>
#include <string>


using namespace std;

OUTPUT_NUMBER_TYPE *init_aggregation_weighted(e_role role, const std::string &address, uint16_t port, seclvl seclvl,
                                     e_mt_gen_alg mt_alg, NUMBER_TYPE *global_model,
                                     vector<NUMBER_TYPE *> *local_models, uint32_t number_of_entries, NUMBER_TYPE *weights);

OUTPUT_NUMBER_TYPE *init_aggregation_normal_avg(e_role role, const std::string &address, uint16_t port, seclvl seclvl,
                                     e_mt_gen_alg mt_alg, NUMBER_TYPE *global_model,
                                     vector<NUMBER_TYPE *> *local_models, uint32_t number_of_entries);
OUTPUT_NUMBER_TYPE *init_aggregation_q_fed(e_role role, const std::string &address, uint16_t port, seclvl seclvl,
                                     e_mt_gen_alg mt_alg, NUMBER_TYPE *global_model,
                                     vector<NUMBER_TYPE *> *deltas, vector<NUMBER_TYPE *> *h_values, uint32_t number_of_entries);
                                     

#endif /* __MPC_AGGREGATOR_H_ */


