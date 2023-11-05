#include <vector>

ArithmeticShare perform_division(uint32_t bitlen, size_t number_of_elements, TypedArithmeticCircuit &ac,
                                 TypedYaoCircuit &yc, size_t number_of_models, ArithmeticShare &summed_updates) {
    size_t boundary_for_negative_values = (1U << (bitlen - 1)) - 1;
    auto boundary_for_negative_values_a = ac->PutSIMDCONSGate(number_of_elements, boundary_for_negative_values, bitlen);
    auto inverse_update = ac->PutINVGate(summed_updates);
    auto is_negative = ac->PutGTGate(summed_updates, boundary_for_negative_values_a);
    auto abs_values = ac->PutMUXGate(inverse_update, summed_updates, is_negative);

    YaoShare summed_abs_updates_y = yc->PutA2YGate(abs_values);
    auto divisor = yc->PutSIMDCONSGate(number_of_elements, number_of_models, bitlen);
    auto aggregated_abs_update = yc->divide(summed_abs_updates_y, divisor);
    auto aggregated_abs_update_a = ac->PutY2AGate(aggregated_abs_update);
    auto inverse_aggregated_update = ac->PutINVGate(aggregated_abs_update_a);
    return ac->PutMUXGate(inverse_aggregated_update, aggregated_abs_update_a, is_negative);
}

OUTPUT_NUMBER_TYPE *aggregate_models(MPCParty &party, uint32_t bitlen, size_t number_of_elements,
                                     vector<ArithmeticShare> *updates, TypedArithmeticCircuit &ac, TypedYaoCircuit &yc,
                                     const ArithmeticShare &global_model, const ArithmeticShare * q_vals) {
    ArithmeticShare summed_updates = ac->createDummyShare();
    bool is_first_update = true;

    int i = 0;

    for (const auto &update : *updates)
    {
        ArithmeticShare curr = ac->PutMULGate(update, q_vals[i++]);
        if (is_first_update) {
            summed_updates = curr;
            is_first_update = false;
        } else {
            summed_updates = ac->PutADDGate(summed_updates, curr);
        }
    }
    ArithmeticShare aggregated_update = perform_division(bitlen, number_of_elements, ac, yc, updates->size(),
                                                         summed_updates);
    ArithmeticShare aggregated_model = ac->PutADDGate(aggregated_update, global_model);
    UntypedSharedOutputShare output_share = ac->PutSharedOUTGate(aggregated_model);
    party->ExecCircuit("Aggregation");
    uint32_t actual_bitlen, actual_number_of_elements;
    OUTPUT_NUMBER_TYPE *result_values;
    output_share->get_shared_value_vec(&result_values, &actual_bitlen, &actual_number_of_elements);
    assert(bitlen == actual_bitlen);
    assert(number_of_elements == actual_number_of_elements);
    return result_values;
}
