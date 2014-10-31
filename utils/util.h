#ifndef __UTIL_H__
#define __UTIL_H__
#include <iostream>
#include <string>
#include <ctime>
#include <boost/shared_ptr.hpp>
#include <Eigen/Dense>
#include <Eigen/Sparse>

class timeutil {
public:
	timeutil();
	void tic();
	double toc();

private:
	clock_t t_;
};

class LbfgsRet {
public:
	static std::string Error(int errorcode);
};

class PRUtil {
public:
	static void score(Eigen::VectorXd& parameter,
		Eigen::SparseMatrix<double, Eigen::RowMajor>& data, Eigen::VectorXi& label,
		double& precision, double& recall);
};

#endif