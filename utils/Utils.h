#ifndef __UTILS_H_
#define __UTILS_H_
	
#include <list>
#include "abycore/circuit/share.h"
#include <string>

using namespace std;

string get_time_as_string();

void print_stack_trace();

void seed_random_generator();
	
#endif /* __UTILS_H_ */

