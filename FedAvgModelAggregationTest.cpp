#include <iostream>
#include <cassert>
#include <vector>
#include <csignal>
#include <ENCRYPTO_utils/crypto/crypto.h>
#include <ENCRYPTO_utils/parse_options.h>
#include <abycore/aby/abyparty.h>
#include "utils/Utils.h"
#include "utils/Constants.h"
#include "mpcutils/MPCConstants.h"
#include "utils/ClientServerConnector.h"
#include "mpcaggregator/MPCAggregator.h"

using namespace std;

void handler(int sig)
{
    cerr << get_time_as_string() << "Error: signal " << sig << endl;
    print_stack_trace();
    exit(1);
}

int32_t read_test_options(int32_t *argcp, char ***argvp, e_role *role,
                          uint32_t *bitlen, uint32_t *secparam, std::string *address,
                          uint16_t *port, string *dataset, size_t *n_models, size_t *m_parameters)
{

    uint32_t int_role = 0, int_port = 0;

    parsing_ctx options[] =
        {{(void *)&int_role, T_NUM, "r", "Role: 0/1", true, false},
         {(void *)n_models, T_NUM, "n", "Max Number of models", true, false},
         {(void *)m_parameters, T_NUM, "m", "Number of parameters", false, false},
         {(void *)bitlen, T_NUM, "b", "Bit-length, default 32", false, false},
         {(void *)secparam, T_NUM, "s", "Symmetric Security Bits, default: 128", false, false},
         {(void *)address, T_STR, "a", "IP-address, default: localhost", false, false},
         {(void *)dataset, T_STR, "d", "dataset name", false, false},
         {(void *)&int_port, T_NUM, "p", "Port, default: 7766", false, false}};

    if (!parse_options(argcp, argvp, options,
                       sizeof(options) / sizeof(parsing_ctx)))
    {
        print_usage(*argvp[0], options, sizeof(options) / sizeof(parsing_ctx));
        std::cout << "Exiting" << std::endl;
        exit(0);
    }

    assert(int_role < 2);
    *role = (e_role)int_role;

    if (int_port != 0)
    {
        assert(int_port < 1U << (sizeof(uint16_t) * 8));
        *port = (uint16_t)int_port;
    }

    // delete options;

    return 1;
}

int main(int argc, char **argv)
{
    signal(SIGSEGV, handler);
    e_role mpc_role;
    uint32_t bitlen = 32, secparam = 128;
    uint16_t port = 7766;
    string address = "127.0.0.1";
    e_mt_gen_alg mt_alg = MT_OT;
    string dataset = "TestData";
    size_t n = 100;
    size_t real_parameters = -1;

    read_test_options(&argc, &argv, &mpc_role, &bitlen, &secparam, &address, &port, &dataset, &n, &real_parameters);

    seclvl seclvl = get_sec_lvl(secparam);

    ROLE_TYPE role = mpc_role ? CLIENT_KEY : SERVER_KEY;
    cout << get_time_as_string() << "This is Aggregator" << role << endl;
    cout << "(Is server: " << (mpc_role == SERVER) << ")" << endl;
    string directory = DATA_DIR + dataset + "Splits/";
    auto [entries_per_model, models_of_client] = read_local_models(directory, role, n);

    assert(entries_per_model > 0);
    assert(!models_of_client->empty());
    assert(models_of_client->size() <= n);
    NUMBER_TYPE *global_model = read_global_model(directory, entries_per_model);
    NUMBER_TYPE *q_vals = read_q_vals(directory, entries_per_model);

    if (real_parameters <= 0)
    {
        cout << "WARNING: USE REDUCED NUMBER OF PARAMETERS!!! (" << real_parameters << " instead of "
             << entries_per_model
             << ")" << endl;
        entries_per_model = real_parameters;
    }
    // TODO insert all q_values into 1 vector of type NUMBER_TYPE

    OUTPUT_NUMBER_TYPE *aggregated_model = aggregate_models(mpc_role, address, port, seclvl, mt_alg, global_model,
                                                            models_of_client, entries_per_model, q_vals);
    cout << get_time_as_string() << "Aggregation Component is Done" << endl;

    send_aggregated_model(DATA_DIR + "Aggregated/", entries_per_model, aggregated_model,
                          role);

    delete[] (*models_of_client)[0];
    models_of_client->clear();
    delete models_of_client;
    delete[] aggregated_model;
    delete[] global_model;

    cout << get_time_as_string() << "Good Night!" << endl;
    return 0;
}
