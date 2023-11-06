#include <vector>

/// @brief performs a division over two arithmetic shares
/// @param bitlen the bitlength of the shares. 
/// @param number_of_elements the number of elements each share has. 
/// @param ac the arithmetic circuit
/// @param yc the yaocircuit
/// @param dividend the divident of the division
/// @param divisor the divisor of the division
/// @return the quotient of the division
ArithmeticShare perform_division(uint32_t bitlen, size_t number_of_elements, TypedArithmeticCircuit &ac,
                                 TypedYaoCircuit &yc, const ArithmeticShare &dividend, const ArithmeticShare &divisor) {
    size_t boundary_for_negative_values = (1U << (bitlen - 1)) - 1;
    auto boundary_for_negative_values_a = ac->PutSIMDCONSGate(number_of_elements, boundary_for_negative_values, bitlen);
    auto inverse_update = ac->PutINVGate(dividend);
    auto is_negative = ac->PutGTGate(dividend, boundary_for_negative_values_a);
    auto abs_values = ac->PutMUXGate(inverse_update, dividend, is_negative);

    YaoShare dividend_y = yc->PutA2YGate(abs_values);
    auto divisor_y = yc->PutA2YGate(divisor);
    auto quotient_y = yc->divide(dividend_y, divisor_y);
    auto quotient_a = ac->PutY2AGate(quotient_y);
    auto inverse_quotient = ac->PutINVGate(quotient_a);
    return ac->PutMUXGate(inverse_quotient, quotient_a, is_negative);
}
/// @brief performs the 
/// @param party 
/// @param bitlen 
/// @param number_of_elements 
/// @param updates 
/// @param ac 
/// @param yc 
/// @param global_model 
/// @param q_vals 
/// @return 
OUTPUT_NUMBER_TYPE *aggregate_models(MPCParty &party, uint32_t bitlen, size_t number_of_elements,
                                     vector<ArithmeticShare> *updates, TypedArithmeticCircuit &ac, TypedYaoCircuit &yc,
                                     const ArithmeticShare &global_model, const ArithmeticShare * q_vals) {
    ArithmeticShare weighted_average = weighted_average_over_updates(bitlen, number_of_elements, updates, ac, yc, q_vals);
    ArithmeticShare aggregated_model = ac->PutADDGate(weighted_average, global_model);
    UntypedSharedOutputShare output_share = ac->PutSharedOUTGate(aggregated_model);
    party->ExecCircuit("Aggregation");
    uint32_t actual_bitlen, actual_number_of_elements;
    OUTPUT_NUMBER_TYPE *result_values;
    output_share->get_shared_value_vec(&result_values, &actual_bitlen, &actual_number_of_elements);
    assert(bitlen == actual_bitlen);
    assert(number_of_elements == actual_number_of_elements);
    return result_values;
}

ArithmeticShare weighted_average_over_updates(uint32_t bitlen, size_t number_of_elements, vector<ArithmeticShare> *updates, TypedArithmeticCircuit &ac,
TypedYaoCircuit &yc, const ArithmeticShare * q_vals){
    ArithmeticShare summed_updates = ac->createDummyShare();
    bool is_first_update = true;

    int i = 0;
    for (const auto &update : *updates)
    {
        ArithmeticShare curr = perform_division(bitlen,number_of_elements,ac, yc, update, q_vals[i++]); 
        if (is_first_update) {
            summed_updates = curr; 
            is_first_update = false;
        } else {
            summed_updates = ac->PutADDGate(summed_updates, curr); 
        }
    }
    return summed_updates;
}

ArithmeticShare average_over_updates(uint32_t bitlen, size_t number_of_elements, vector<ArithmeticShare> *updates, TypedArithmeticCircuit &ac,
TypedYaoCircuit &yc){
    ArithmeticShare summed_updates = ac->createDummyShare();
    bool is_first_update = true;

    for (const auto &update : *updates)
    {
        if (is_first_update) {
            summed_updates = update; 
            is_first_update = false;
        } else {
            summed_updates = ac->PutADDGate(summed_updates, update); 
        }
    }
    ArithmeticShare aggregated_update = perform_division(bitlen, number_of_elements, ac, yc, updates->size(),
                                                     summed_updates);
    return aggregated_update;
}
