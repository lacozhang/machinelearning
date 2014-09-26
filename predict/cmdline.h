#ifndef __CMD_LINE_H__
#define __CMD_LINE_H__
#include<string>
#include <boost/program_options.hpp>

void parse_cmd(int argc, char* argv[], std::string& modelpath, std::string& input, std::string& output);

#endif // __CMD_LINE_H__