
#ifndef __PARAMETER_H__
#define __PARAMETER_H__
#include <iostream>
#include <string>
#include <vector>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <boost/program_options.hpp>

struct CmdParameters {

	CmdParameters();

	// regularization parameters
	double m_shrink, m_alpha, m_lambda;
	// train file used to adapt parameters
	// test file used to evaluate result
	std::string m_trainFile, m_existModel, m_output;
	std::vector<std::string> m_listTestFiles;
};

struct TestPair {
	std::string setname;
	boost::shared_ptr< Eigen::SparseMatrix<double, Eigen::RowMajor> > samples;
	boost::shared_ptr< Eigen::VectorXi > labels;
};

void parsecmd(int argc, char* argv[], CmdParameters& param);

#endif