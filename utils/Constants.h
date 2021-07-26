#ifndef __CONSTANTS_H_
#define __CONSTANTS_H_
	
#include <cstdint>
#include <string>

static const std::string DATA_DIR = "../data/";
typedef uint64_t NUMBER_TYPE;
typedef int64_t SIGNED_NUMBER_TYPE;
typedef NUMBER_TYPE OUTPUT_NUMBER_TYPE;
typedef SIGNED_NUMBER_TYPE SIGNED_OUTPUT_NUMBER_TYPE;
typedef std::string ROLE_TYPE;
static const ROLE_TYPE SERVER_KEY = "A";
static const ROLE_TYPE CLIENT_KEY = "B";
	
#endif /* __CONSTANTS_H_ */
