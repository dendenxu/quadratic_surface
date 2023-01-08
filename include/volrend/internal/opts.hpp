#pragma once
// Command line option parsing

#include <cxxopts.hpp>

namespace volrend {
namespace internal {

void add_common_opts(cxxopts::Options& options);

cxxopts::ParseResult parse_options(cxxopts::Options& options, int argc,
                                   char* argv[]);

}  // namespace internal
}  // namespace volrend
