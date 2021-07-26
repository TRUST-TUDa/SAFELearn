#include "abycore/aby/abyparty.h"
#include <memory>
#include "MPCIterationCounter.h"
#include "MPCCircuit.h"
#include "../utils/Utils.h"
#include <iostream>

#ifndef FLGUARDMPC_MPCPARTY_H
#define FLGUARDMPC_MPCPARTY_H

using namespace std;

class MPCParty_ {

private:
    ABYParty *actual_party;
    IterationCounter_ *iterationCounter;
    TypedBooleanCircuit boolean_circuit;
    TypedYaoCircuit yao_circuit;
    TypedArithmeticCircuit arithmetic_circuit;

public:
    MPCParty_(e_role role, const string &address, uint16_t port, seclvl seclvl,
              uint32_t bitlen, uint32_t nthreads, e_mt_gen_alg mt_alg) {
        iterationCounter = new IterationCounter_();
        actual_party = new ABYParty(role, address, port, seclvl, bitlen, nthreads, mt_alg);
        std::vector<Sharing *> &sharings = actual_party->GetSharings();

        ArithmeticCircuit *arith_circ = (ArithmeticCircuit *) sharings[S_ARITH]->GetCircuitBuildRoutine();
        BooleanCircuit *y_circ = (BooleanCircuit *) sharings[S_YAO]->GetCircuitBuildRoutine();
        BooleanCircuit *bool_circ = (BooleanCircuit *) sharings[S_BOOL]->GetCircuitBuildRoutine();
        boolean_circuit = new TypedBooleanCircuit_(bool_circ, y_circ, DO_PRINTS, iterationCounter);
        yao_circuit = new TypedYaoCircuit_(y_circ, DO_PRINTS, iterationCounter);
        arithmetic_circuit = new TypedArithmeticCircuit_(arith_circ, boolean_circuit, DO_PRINTS, iterationCounter);
    }

    ~MPCParty_() {
        delete boolean_circuit;
        delete yao_circuit;
        delete arithmetic_circuit;
        delete actual_party;
        delete iterationCounter;
    }

    void Reset() {
        iterationCounter->new_iteration();
        actual_party->Reset();
    }

    void ExecCircuit(string name = "") {
        if (name.empty()) {
            name = to_string(iterationCounter->get_current_state());
        } else {
            name += " (" + to_string(iterationCounter->get_current_state()) + ")";
        }
        cout << get_time_as_string() << "GO: " << name << endl;
        actual_party->ExecCircuit();
        cout << get_time_as_string() << "Done: " << name << endl;
    }

    TypedYaoCircuit get_yao_circuit() {
        return yao_circuit;
    }

    TypedArithmeticCircuit get_arithmetic_circuit() {
        return arithmetic_circuit;
    }

    ITERATION_COUNTER_TYPE get_current_iteration_counter() {
        return iterationCounter->get_current_state();
    }

};


typedef std::unique_ptr<MPCParty_> MPCParty;

#endif //FLGUARDMPC_MPCPARTY_H
