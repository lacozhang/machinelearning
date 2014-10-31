#ifndef __PARAMETER_H__
#define __PARAMETER_H__

#include <string>
#include <boost/shared_ptr.hpp>
#include <Eigen/Dense>
#include <Eigen/Sparse>

typedef struct derivative_parameter {

	/* initialize the parameter for optimization */
	derivative_parameter(){
		l2start = l2end = 0;
		l1start = l1end = 0;
		l2c = 1.0f;
		featfile = outmodel = "";
		featsize = 0;
		iter = 0;
	}

	int l2start, l2end;
	int l1start, l1end;
	int iter;
	double l2c;
	std::string featfile, outmodel;
	int featsize;
} derivative_parameter_t;

typedef struct optimization_instance {
	int poslabel_;
	boost::shared_ptr<Eigen::SparseMatrix<double, Eigen::RowMajor> >& samples_;
	boost::shared_ptr<Eigen::VectorXi>& labels_;
	boost::shared_ptr<derivative_parameter_t>& prog_params_;

	optimization_instance(int poslabel,
	boost::shared_ptr<Eigen::SparseMatrix<double, Eigen::RowMajor> >& samples,
	boost::shared_ptr<Eigen::VectorXi>& labels, 
	boost::shared_ptr<derivative_parameter_t>& prog_params): poslabel_(poslabel),
	samples_(samples), labels_(labels), prog_params_(prog_params)
	{
	}

} optimization_instance_t;

#endif