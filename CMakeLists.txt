
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
cmake_minimum_required(VERSION 3.13)
project(SafeLearn LANGUAGES CXX)

set(ABY_PATH "ABSOLUTE_PATH_TO_ABY")


find_package(GMP REQUIRED)
find_package(Threads REQUIRED)
find_package(Boost 1.66.0 REQUIRED COMPONENTS thread system)
set(MPCAggregationFiles "mpcaggregator/MPCAggregator.cpp")
set(NonMPCRelatedUtilsFile "utils/ClientServerConnector.cpp" "utils/Utils.cpp")

set(NonMainCPPFiles ${MPCAggregationFiles} ${NonMPCRelatedUtilsFile})
set(ABY_LIB_PATH "${ABY_PATH}/lib")
set(NON_ABY_LIBRARIES gmpxx gmp pthread boost_system crypto dl backtrace)
set(ABY_LIBRARIES "${ABY_LIB_PATH}/libaby.a" "${ABY_LIB_PATH}/libencrypto_utils.a" "${ABY_LIB_PATH}/libotextension.a" "${ABY_LIB_PATH}/librelic_s.a")
set(COMPILE_OPTIONS PUBLIC -Wall -Wextra -g3 -std=c++17)

add_executable(fedavg_aggregation FedAvgModelAggregation.cpp ${NonMainCPPFiles})
include_directories(fedavg_aggregation "${ABY_PATH}/include")
target_link_libraries(fedavg_aggregation ${ABY_LIBRARIES})
target_link_libraries(fedavg_aggregation ${NON_ABY_LIBRARIES})
target_compile_options(fedavg_aggregation ${COMPILE_OPTIONS})

set(CMAKE_CXX_STANDARD 17)
