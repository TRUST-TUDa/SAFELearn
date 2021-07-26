#ifndef __MPC_SHARE_H_
#define __MPC_SHARE_H_

#include "abycore/circuit/share.h"
#include <memory>
#include <iostream>
#include "../utils/Constants.h"
#include "../utils/Utils.h"


class AbstractTypedShare {
public:
    share *content;
    ITERATION_COUNTER_TYPE circuitCounter;

    AbstractTypedShare(share *content1, int32_t circuitCounter1) {
        content = content1;
        circuitCounter = circuitCounter1;
    }

    ~AbstractTypedShare() {
        delete content;
    }

    void print_statistics() {
        std::cout << "N: " << content->get_nvals() << " Bitlen: " << content->get_bitlength() << " Max Bitlen: "
                  << content->get_max_bitlength() << std::endl;
    }
};

class YaoShare_ : public AbstractTypedShare {

public:
    explicit YaoShare_(share *content1, int32_t circuitCounter1) : AbstractTypedShare(content1, circuitCounter1) {}

};

class BooleanShare_ : public AbstractTypedShare {

public:
    explicit BooleanShare_(share *content1, int32_t circuitCounter1) : AbstractTypedShare(content1, circuitCounter1) {}

};

class ArithmeticShare_ : public AbstractTypedShare {

public:
    explicit ArithmeticShare_(share *content1, int32_t circuitCounter1) : AbstractTypedShare(content1,
                                                                                             circuitCounter1) {}

};

class UntypedOutputShare_ : public AbstractTypedShare {
public:
    explicit UntypedOutputShare_(share *content1) : AbstractTypedShare(content1, -1) {}

    void get_clear_value_vec(OUTPUT_NUMBER_TYPE **vec, uint32_t *bitlen, uint32_t *nvals) {
        content->get_clear_value_vec(vec, bitlen, nvals);
    }

    template<class T>
    T get_clear_value() {
        if (content->get_nvals() != 1) print_stack_trace();
        assert(content->get_nvals() == 1);
        return content->get_clear_value<T>();
    }
};

class UntypedSharedOutputShare_ : public AbstractTypedShare {
public:
    explicit UntypedSharedOutputShare_(share *content1) : AbstractTypedShare(content1, -1) {}

    template<class T>
    void get_shared_value_vec(T **vec, uint32_t *bitlen, uint32_t *nvals) {
        content->get_clear_value_vec(vec, bitlen, nvals);
    }

    template<class T>
    T get_shared_value() {
        if (content->get_nvals() != 1) print_stack_trace();
        assert(content->get_nvals() == 1);
        return content->get_clear_value<T>();
    }


};

typedef std::shared_ptr<YaoShare_> YaoShare;
typedef std::shared_ptr<BooleanShare_> BooleanShare;
typedef std::shared_ptr<ArithmeticShare_> ArithmeticShare;
typedef std::shared_ptr<UntypedOutputShare_> UntypedOutputShare;
typedef std::shared_ptr<UntypedSharedOutputShare_> UntypedSharedOutputShare;

#endif /* __MPC_SHARE_H_ */
