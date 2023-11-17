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

/// @brief calculates the weighted average over all the local_updates according to the weights (q_values)
/// @param bitlen bitlen of the shares
/// @param number_of_elements the number of elements each share has
/// @param local_updates a vector of all local_updates as arithmetic shares
/// @param ac the arithmetic circuit
/// @param yc the yao circuit
/// @param q_vals an array of the weights as arithmetic shares for each update
/// @return the weighted average over all local_updates according to the weights. 
ArithmeticShare weighted_average_over_updates(uint32_t bitlen, size_t number_of_elements, vector<ArithmeticShare> *local_updates, TypedArithmeticCircuit &ac,
TypedYaoCircuit &yc, const ArithmeticShare * q_vals){
    ArithmeticShare summed_updates = ac->createDummyShare();
    bool is_first_update = true;

    int i = 0;
    for (const auto &update : *local_updates)
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

/// @brief calculates the normal unweighted average over all update shares
/// @param bitlen the bitlen of the shares
/// @param number_of_elements the number of element each share has
/// @param local_updates a vector of all local_updates as arithmetic shares
/// @param ac the arithmetic circuit
/// @param yc the yao circuit
/// @return the unweighted average over all update shares. 
ArithmeticShare average_over_updates(uint32_t bitlen, size_t number_of_elements, vector<ArithmeticShare> *local_updates, TypedArithmeticCircuit &ac,
TypedYaoCircuit &yc){
    ArithmeticShare summed_updates = ac->createDummyShare();
    bool is_first_update = true;

    for (const auto &update : *local_updates)
    {
        if (is_first_update) {
            summed_updates = update; 
            is_first_update = false;
        } else {
            summed_updates = ac->PutADDGate(summed_updates, update); 
        }
    }

    ArithmeticShare divisor = ac->PutSIMDCONSGate(number_of_elements, local_updates->size(), bitlen);
    ArithmeticShare aggregated_update = perform_division(bitlen, number_of_elements, ac, yc, summed_updates, divisor);
    return aggregated_update;
}

ArithmeticShare q_fed_over_updates(uint32_t bitlen, size_t number_of_elements, vector<ArithmeticShare> *deltas, vector<ArithmeticShare> *h, TypedArithmeticCircuit &ac,
                                TypedYaoCircuit &yc, const ArithmeticShare &global_model){
    ArithmeticShare summed_deltas = ac->createDummyShare();
    ArithmeticShare summed_h = ac->createDummyShare();
    bool is_first_update = true;


    for (const auto &update : *deltas)
    {
        if (is_first_update) {
            summed_deltas = update; 
            is_first_update = false;
        } else {
            summed_deltas = ac->PutADDGate(summed_deltas, update); 
        }
    }
    
    bool is_first_update = true;
    for (const auto &update : *h)
    {
        if (is_first_update) {
            summed_h = update; 
            is_first_update = false;
        } else {
            summed_h = ac->PutADDGate(summed_h, update); 
        }
    }
    
    ArithmeticShare division = perform_division(bitlen, number_of_elements, ac, yc, summed_deltas, summed_h);
    ArithmeticShare aggregated_update = ac->PutSUBGate(global_model, division);
    return aggregated_update;
}


/// @brief local_updates 
/// @param party the mpc party 
/// @param bitlen the bitlen of the shares
/// @param number_of_elements the number of element each share has
/// @param local_updates a vector of all local_updates as arithmetic shares
/// @param ac the arithmetic circuit
/// @param yc the yao circuit
/// @param global_model the global model 
/// @param q_vals the weights (used for weighted average)
/// @return The update as OUTPUT_NUMBER_TYPE
OUTPUT_NUMBER_TYPE *aggregate_models_weighted(MPCParty &party, uint32_t bitlen, size_t number_of_elements,
                                     vector<ArithmeticShare> *local_updates, TypedArithmeticCircuit &ac, TypedYaoCircuit &yc,
                                     const ArithmeticShare &global_model, const ArithmeticShare * q_vals) {
    ArithmeticShare average = weighted_average_over_updates(bitlen, number_of_elements, local_updates, ac, yc, q_vals);
    //ArithmeticShare average = average_over_updates(bitlen, number_of_elements, local_updates, ac, yc); -- use this if normal average is needed instead of weighted one
    ArithmeticShare aggregated_model = ac->PutADDGate(average, global_model);
    UntypedSharedOutputShare output_share = ac->PutSharedOUTGate(aggregated_model);
    party->ExecCircuit("Aggregation");
    uint32_t actual_bitlen, actual_number_of_elements;
    OUTPUT_NUMBER_TYPE *result_values;
    output_share->get_shared_value_vec(&result_values, &actual_bitlen, &actual_number_of_elements);
    assert(bitlen == actual_bitlen);
    assert(number_of_elements == actual_number_of_elements);
    return result_values;
}

/// @brief local_updates 
/// @param party the mpc party 
/// @param bitlen the bitlen of the shares
/// @param number_of_elements the number of element each share has
/// @param local_updates a vector of all local_updates as arithmetic shares
/// @param ac the arithmetic circuit
/// @param yc the yao circuit
/// @param global_model the global model 
/// @param q_vals the weights (used for weighted average)
/// @return The update as OUTPUT_NUMBER_TYPE
OUTPUT_NUMBER_TYPE *aggregate_models_normal_avg(MPCParty &party, uint32_t bitlen, size_t number_of_elements,
                                     vector<ArithmeticShare> *local_updates, TypedArithmeticCircuit &ac, TypedYaoCircuit &yc,
                                     const ArithmeticShare &global_model) {
    ArithmeticShare average = average_over_updates(bitlen, number_of_elements, local_updates, ac, yc); 
    ArithmeticShare aggregated_model = ac->PutADDGate(average, global_model);
    UntypedSharedOutputShare output_share = ac->PutSharedOUTGate(aggregated_model);
    party->ExecCircuit("Aggregation");
    uint32_t actual_bitlen, actual_number_of_elements;
    OUTPUT_NUMBER_TYPE *result_values;
    output_share->get_shared_value_vec(&result_values, &actual_bitlen, &actual_number_of_elements);
    assert(bitlen == actual_bitlen);
    assert(number_of_elements == actual_number_of_elements);
    return result_values;
}


OUTPUT_NUMBER_TYPE *aggregate_models_q_fed(MPCParty &party, uint32_t bitlen, size_t number_of_elements,
                                     vector<ArithmeticShare> *deltas, vector<ArithmeticShare> * h_vec, 
                                     TypedArithmeticCircuit &ac, TypedYaoCircuit &yc, const ArithmeticShare &global_model)
    {
    ArithmeticShare aggregated_model = q_fed_over_updates(bitlen, number_of_elements, deltas, h_vec, ac, yc, global_model);
    UntypedSharedOutputShare output_share = ac->PutSharedOUTGate(aggregated_model);
    party->ExecCircuit("Aggregation");
    uint32_t actual_bitlen, actual_number_of_elements;
    OUTPUT_NUMBER_TYPE *result_values;
    output_share->get_shared_value_vec(&result_values, &actual_bitlen, &actual_number_of_elements);
    assert(bitlen == actual_bitlen);
    assert(number_of_elements == actual_number_of_elements);
    return result_values;
}