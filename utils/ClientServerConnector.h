#ifndef __CLIENT_SERVER_CONNECTOR_H_
#define __CLIENT_SERVER_CONNECTOR_H_

#include "Constants.h"
#include <vector>
#include <tuple>
#include <string>

using namespace std;

tuple<uint32_t, vector<NUMBER_TYPE *> *> read_local_models(string directory, ROLE_TYPE role, size_t max_models_to_read);

NUMBER_TYPE *read_global_model(string directory, uint32_t number_of_entries);
NUMBER_TYPE *read_q_vals(string directory, uint32_t number_of_entries);

void send_aggregated_model(string directory, uint32_t number_of_entries_per_model, OUTPUT_NUMBER_TYPE *model,
                           ROLE_TYPE role);

#endif /* __CLIENT_SERVER_CONNECTOR_H_ */
