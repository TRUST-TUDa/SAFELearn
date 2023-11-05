#ifndef __MPC_CIRCUIT_H_
#define __MPC_CIRCUIT_H_

#include <memory>
#include "MPCShare.h"
#include "abycore/circuit/booleancircuits.h"
#include "abycore/sharing/sharing.h"
#include "abycore/circuit/arithmeticcircuits.h"
#include <vector>
#include <iostream>
#include "MPCIterationCounter.h"
#include <abycore/sharing/boolsharing.h>
#include "../utils/Constants.h"
#include "../utils/Utils.h"
#include <map>

using namespace std;

template <typename CircuitClass, typename SharingClass_, typename SharingClass>
class AbstractTypedCircuit
{

protected:
    CircuitClass *actual_circuit;
    bool do_prints;
    vector<share *> print_gates;
    string name;
    IterationCounter iterationCounter;
    map<size_t, uint32_t *> odd_subsets, even_subsets;
    uint32_t expected_bitlength;

    void do_qa(SharingClass result, uint32_t expected_values)
    {
        // Unfortunately, ABY don't care about setting maxbitlen correctly. Therefore, we can not check it here
        assert(result->content->get_nvals() == expected_values);
    }

    void update_max_length(share *s)
    {
        if (expected_bitlength == 1 && s->get_max_bitlength() < BIT_LENGTH)
        {
            s->set_max_bitlength(BIT_LENGTH);
        }
    }

    SharingClass create_new_typed_share(share *to_convert)
    {
        SharingClass result = make_shared<SharingClass_>(to_convert, iterationCounter->get_current_state());
        return result;
    }

public:
    AbstractTypedCircuit(CircuitClass *actual_circuit1, bool do_prints1, const string &name1,
                         IterationCounter iterationCounter1, uint32_t expected_bitlength1)
    {
        actual_circuit = actual_circuit1;
        do_prints = do_prints1;
        expected_bitlength = expected_bitlength1;
        iterationCounter = iterationCounter1;
        name = name1;
    }

    ~AbstractTypedCircuit()
    {
        std::cout << name << "Circuit Destroyed\n";
        for_each(print_gates.begin(), print_gates.end(), [](share *s)
                 { delete s; });
        for (auto map_entry : odd_subsets)
        {
            delete[] get<1>(map_entry);
        }
        for (auto map_entry : even_subsets)
        {
            delete[] get<1>(map_entry);
        }
    }

    template <typename T>
    bool is_current_circuit_iteration(T s)
    {
        return s->circuitCounter == iterationCounter->get_current_state();
    }

    template <typename T>
    void enforce_current_circuit_iteration(T s)
    {
        // ensure, that the party was not reset since this share was created
        if (!is_current_circuit_iteration(s))
        {
            cout << "Got old share, expected iteration " << iterationCounter->get_current_state() << " but got "
                 << s->circuitCounter << endl;
            print_stack_trace();
        }
        assert(s->circuitCounter == iterationCounter->get_current_state());
    }

    void enforce_simd_compability(const SharingClass &s1, const SharingClass &s2)
    {
        enforce_current_circuit_iteration(s1);
        enforce_current_circuit_iteration(s2);
        if (s1->content->get_nvals() != s2->content->get_nvals())
        {
            std::cout << s1->content->get_nvals() << " vs. " << s2->content->get_nvals() << std::endl;
            print_stack_trace();
        }
        assert(s1->content->get_nvals() == s2->content->get_nvals());
    }

    void enforce_compability(const SharingClass &s1, const SharingClass &s2)
    {
        if (s1->content->get_bitlength() != s2->content->get_bitlength())
            print_stack_trace();
        assert(s1->content->get_bitlength() == s2->content->get_bitlength());
        enforce_simd_compability(s1, s2);
    }

    SharingClass PutADDGate(const SharingClass &s1, const SharingClass &s2)
    {
        enforce_compability(s1, s2);
        share *result = actual_circuit->PutADDGate(s1->content, s2->content);
        update_max_length(result);
        SharingClass typed_result = create_new_typed_share(result);
        do_qa(typed_result, s1->content->get_nvals());
        return typed_result;
    }

    SharingClass PutMULGate(const SharingClass &s1, const SharingClass &s2)
    {
        enforce_compability(s1, s2);
        share *result = actual_circuit->PutMULGate(s1->content, s2->content);
        update_max_length(result);
        SharingClass typed_result = create_new_typed_share(result);
        do_qa(typed_result, s1->content->get_nvals());
        return typed_result;
    }

    void PutPrintValueGate(SharingClass s, const std::string &print_name)
    {
        enforce_current_circuit_iteration(s);
        if (do_prints)
        {
            share *print_share = actual_circuit->PutPrintValueGate(s->content, print_name.c_str());
            delete print_share;
        }
    }

    SharingClass PutSIMDCONSGate(uint32_t nvals, NUMBER_TYPE val, uint32_t bitlen)
    {
        share *result = actual_circuit->PutSIMDCONSGate(nvals, val, bitlen);
        SharingClass typed_result = create_new_typed_share(result);
        do_qa(typed_result, nvals);
        return typed_result;
    }

    template <typename T>
    SharingClass PutSharedSIMDINGate(uint32_t number_of_entries, T *input, uint32_t bitlen)
    {
        share *s = actual_circuit->PutSharedSIMDINGate(number_of_entries, input, bitlen);
        if (expected_bitlength == 1)
        {
            s->set_max_bitlength(BIT_LENGTH);
        }
        SharingClass typed_result = create_new_typed_share(s);
        do_qa(typed_result, number_of_entries);
        return typed_result;
    }

    SharingClass PutSIMDINGate(uint32_t number_of_entries, NUMBER_TYPE *input, uint32_t bitlen, e_role role)
    {
        share *s = actual_circuit->PutSIMDINGate(number_of_entries, input, bitlen, role);
        if (expected_bitlength == 1)
        {
            s->set_max_bitlength(BIT_LENGTH);
        }
        SharingClass typed_result = create_new_typed_share(s);
        do_qa(typed_result, number_of_entries);
        return typed_result;
    }

    SharingClass createDummyShare()
    {
        return make_shared<SharingClass_>(nullptr, -1);
    }
};

class TypedBooleanCircuit_ : public AbstractTypedCircuit<BooleanCircuit, BooleanShare_, BooleanShare>
{
private:
    BooleanCircuit *yao_circuit;

public:
    explicit TypedBooleanCircuit_(BooleanCircuit *circuit1, BooleanCircuit *yao_circuit1, bool do_prints,
                                  IterationCounter iterationCounter1) :

                                                                        AbstractTypedCircuit(circuit1, do_prints, "Boolean", iterationCounter1, BIT_LENGTH)
    {
        yao_circuit = yao_circuit1;
    }

    BooleanShare PutY2BGate(const YaoShare &s)
    {
        enforce_current_circuit_iteration(s);
        share *res = actual_circuit->PutY2BGate(s->content);
        BooleanShare typed_result = create_new_typed_share(res);
        do_qa(typed_result, s->content->get_nvals());
        return typed_result;
    }

    BooleanShare PutA2BGate(const ArithmeticShare &s)
    {
        enforce_current_circuit_iteration(s);
        share *s_y = yao_circuit->PutA2YGate(s->content);
        share *res = actual_circuit->PutY2BGate(s_y);
        delete s_y;
        BooleanShare typed_result = create_new_typed_share(res);
        do_qa(typed_result, s->content->get_nvals());
        return typed_result;
    }

    BooleanShare PutGTGate(const BooleanShare &s1, const BooleanShare &s2)
    {
        enforce_compability(s1, s2);
        share *result = actual_circuit->PutGTGate(s1->content, s2->content);
        BooleanShare typed_result = create_new_typed_share(result);
        return typed_result;
    }

    BooleanShare PutMUXGate(const BooleanShare &ina, const BooleanShare &inb, const BooleanShare &sel)
    {
        enforce_compability(ina, inb);
        enforce_simd_compability(inb, sel);
        share *result = actual_circuit->PutMUXGate(ina->content, inb->content, sel->content);
        BooleanShare typed_result = create_new_typed_share(result);
        do_qa(typed_result, ina->content->get_nvals());
        return typed_result;
    }
};

static const string BIT_LENGTH_AS_STRING = to_string(BIT_LENGTH);
static const string DIVISION_FILE_NAME = "../div_" + BIT_LENGTH_AS_STRING + ".aby";

typedef TypedBooleanCircuit_ *TypedBooleanCircuit;

class TypedYaoCircuit_ : public AbstractTypedCircuit<BooleanCircuit, YaoShare_, YaoShare>
{
private:
public:
    TypedYaoCircuit_(BooleanCircuit *circuit1, bool do_prints, IterationCounter iterationCounter1)
        : AbstractTypedCircuit(circuit1, do_prints, "Yao", iterationCounter1, BIT_LENGTH) {}

    YaoShare PutA2YGate(const ArithmeticShare &s)
    {
        enforce_current_circuit_iteration(s);
        share *res = actual_circuit->PutA2YGate(s->content);
        YaoShare typed_result = create_new_typed_share(res);
        do_qa(typed_result, s->content->get_nvals());
        return typed_result;
    }

    YaoShare divide(YaoShare &dividend, YaoShare &divisor)
    {
        enforce_compability(dividend, divisor);
        vector<uint32_t> wires = dividend->content->get_wires();
        wires.insert(wires.end(), divisor->content->get_wires().begin(), divisor->content->get_wires().end());
        uint32_t num_simd = dividend->content->get_nvals();
        vector<uint32_t> int_div_share = actual_circuit->PutGateFromFile(DIVISION_FILE_NAME, wires, num_simd);
        int_div_share.pop_back();

        share *out = new boolshare(int_div_share, actual_circuit);
        YaoShare result = create_new_typed_share(out);
        return result;
    }
};

typedef TypedYaoCircuit_ *TypedYaoCircuit;

class TypedArithmeticCircuit_ : public AbstractTypedCircuit<ArithmeticCircuit, ArithmeticShare_, ArithmeticShare>
{
private:
    TypedBooleanCircuit typed_boolean_circuit;

public:
    TypedArithmeticCircuit_(ArithmeticCircuit *circuit1,
                            const TypedBooleanCircuit &typed_boolean_circuit1, bool do_prints,
                            IterationCounter iterationCounter1)
        : AbstractTypedCircuit(circuit1, do_prints, "Arithmetic", iterationCounter1, 1)
    {
        typed_boolean_circuit = typed_boolean_circuit1;
    };

    ArithmeticShare PutY2AGate(YaoShare &s)
    {
        enforce_current_circuit_iteration(s);
        BooleanShare bool_share = typed_boolean_circuit->PutY2BGate(s);
        return PutB2AGate(bool_share);
    }

    ArithmeticShare PutB2AGate(const BooleanShare &s)
    {
        enforce_current_circuit_iteration(s);
        share *res = actual_circuit->PutB2AGate(s->content);
        return create_new_typed_share(res);
    }

    UntypedSharedOutputShare PutSharedOUTGate(const ArithmeticShare &s)
    {
        enforce_current_circuit_iteration(s);
        share *out_share = actual_circuit->PutSharedOUTGate(s->content);
        UntypedSharedOutputShare result = make_shared<UntypedSharedOutputShare_>(out_share);
        return result;
    }

    ArithmeticShare PutGTGate(const ArithmeticShare &s1, const ArithmeticShare &s2)
    {
        enforce_compability(s1, s2);
        BooleanShare s1_b = typed_boolean_circuit->PutA2BGate(s1);
        BooleanShare s2_b = typed_boolean_circuit->PutA2BGate(s2);
        BooleanShare result_b = typed_boolean_circuit->PutGTGate(s1_b, s2_b);
        return PutB2AGate(result_b);
    }

    ArithmeticShare PutMUXGate(const ArithmeticShare &ina, const ArithmeticShare &inb, const ArithmeticShare &sel)
    {
        enforce_compability(ina, inb);
        enforce_simd_compability(inb, sel);
        BooleanShare ina_b = typed_boolean_circuit->PutA2BGate(ina);
        BooleanShare inb_b = typed_boolean_circuit->PutA2BGate(inb);
        BooleanShare sel_b = typed_boolean_circuit->PutA2BGate(sel);

        BooleanShare result_b = typed_boolean_circuit->PutMUXGate(ina_b, inb_b, sel_b);
        return PutB2AGate(result_b);
    }

    ArithmeticShare PutINVGate(const ArithmeticShare &s)
    {
        enforce_current_circuit_iteration(s);
        auto wires = s->content->get_wires();
        assert(wires.size() == 1);
        uint32_t wire_id = wires[0];
        uint32_t result_wire = actual_circuit->PutINVGate(wire_id);
        vector<uint32_t> result_wires(1, result_wire);
        share *result = new arithshare(result_wires, actual_circuit);
        return create_new_typed_share(result);
    }
};

typedef TypedArithmeticCircuit_ *TypedArithmeticCircuit;

#endif /* __MPC_CIRCUIT_H_ */
