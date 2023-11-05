#ifndef __MPC_AGGREGATOR_H_
#define __MPC_AGGREGATOR_H_

#include "../utils/Constants.h"
#include <vector>
#include <tuple>
#include <string>

using namespace std;

OUTPUT_NUMBER_TYPE *aggregate_models(e_role role, const std::string &address, uint16_t port, seclvl seclvl,
                                     e_mt_gen_alg mt_alg, NUMBER_TYPE *global_model,
                                     vector<NUMBER_TYPE *> *local_models, uint32_t number_of_entries, _Float64 *q_vals);

#endif /* __MPC_AGGREGATOR_H_ */
