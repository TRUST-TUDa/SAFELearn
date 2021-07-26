#ifndef FLGUARDMPC_MPCITERATIONCOUNTER_H
#define FLGUARDMPC_MPCITERATIONCOUNTER_H

#include <inttypes.h>
#include <memory>
#include "../utils/Constants.h"


class IterationCounter_ {

private:
    ITERATION_COUNTER_TYPE current_state;
public:
    IterationCounter_() {
        current_state = 0;
    }

    void new_iteration() {
        current_state++;
    }

    ITERATION_COUNTER_TYPE get_current_state() {
        return current_state;
    }

};

typedef IterationCounter_ *IterationCounter;

#endif //FLGUARDMPC_MPCITERATIONCOUNTER_H
