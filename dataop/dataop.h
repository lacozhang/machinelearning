#include <string>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <boost/shared_ptr.hpp>
#include "../utils/util.h"

#ifndef __DATA_OP_H__
#define __DATA_OP_H__

void matrix_size_estimation(std::string featfile, Eigen::VectorXi& datsize, int& row, int& col);

void load_data(std::string featfile,
			   boost::shared_ptr<Eigen::SparseMatrix<double, Eigen::RowMajor> >& Samples,
			   boost::shared_ptr<Eigen::VectorXi>& labels,
			   int prow=-1, int pcols=-1);
#endif