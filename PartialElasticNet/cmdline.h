
#ifndef __CMD_LINE_H__
#define __CMD_LINE_H__

#include <lbfgs.h>
#include "parameters.h"

void parse_command_line(int argc, char* argv[], 
						boost::shared_ptr<lbfgs_parameter_t>& param,
						boost::shared_ptr<derivative_parameter_t>& derivative_param);
#endif