#include "Utils.h"

#define BOOST_STACKTRACE_USE_ADDR2LINE
#define BOOST_STACKTRACE_USE_BACKTRACE
#include <boost/stacktrace.hpp>

string get_time_as_string() {
    time_t rawtime;
    struct tm * timeinfo;
    char buffer[80];
    time(&rawtime);
    timeinfo = localtime(&rawtime);
    strftime(buffer, 80, "%Y-%m-%d %H:%M:%S ", timeinfo);
    return string(buffer);
}

void print_stack_trace(){
    cout << boost::stacktrace::stacktrace() << endl;
}

void seed_random_generator(){
    srand(0);
}
