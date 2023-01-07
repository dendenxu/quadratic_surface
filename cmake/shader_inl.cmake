FILE(WRITE ${OUTPUT} "")
FILE(READ ${SHADER_FILE} contents)
FILE(APPEND ${OUTPUT} "#pragma once\n#include <string>\nnamespace {\nconst std::string ${SHADER_NAME}_SRC =\nR\"glsl(\n")
string(LENGTH "${contents}" len)
message(STATUS "Getting string length ${len}")
set(blk_size 1024)
set(curr 0)
while(${curr} LESS ${len})
    string(SUBSTRING "${contents}" ${curr} ${blk_size} sub) # gettin substring of contents
    # message(STATUS "Getting substring:\n${sub}") # this checks out
    file(APPEND ${OUTPUT} "${sub}") # this doesn't check out
    file(APPEND ${OUTPUT} ")glsl\"\nR\"glsl(") # add a new line
    MATH(EXPR curr "${curr}+${blk_size}")
    message(STATUS "Setting curr to ${curr}")
endwhile()
# ! CMAKE seems to be evaluating comments whatsoever, why?
# FILE(APPEND ${OUTPUT} "${contents}")
FILE(APPEND ${OUTPUT} ")glsl\";\n}  // namespace")
