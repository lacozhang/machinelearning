#include <iostream>
#include <fstream>
#include <string>
#include <hash_set>
#include <Eigen/Sparse>
#include <boost/shared_ptr.hpp>
#include <boost/regex.hpp>
#include <lbfgs.h>
#include "dataop.h"
#include "cmdline.h"
#include "parameters.h"
#include "train.h"

boost::shared_ptr<lbfgs_parameter_t>  lbfgs_param;
boost::shared_ptr<derivative_parameter_t> program_param;
boost::shared_ptr<Eigen::SparseMatrix<double, Eigen::RowMajor> > insts;
boost::shared_ptr<Eigen::VectorXi> clslabels;

int main(int argc, char* argv[]){

	lbfgs_param.reset( new lbfgs_parameter_t() );
	program_param.reset( new derivative_parameter_t() );

	int sample_size = 0, feat_size = 0;

	if( lbfgs_param.get() == NULL ){
		std::cerr << "Error, lbfgs_parameter_t allocation failed" << std::endl;
		std::abort();
	}

	if( program_param.get() == NULL ){
		std::cerr << "Error, derivative_parameter_t allocation failed" << std::endl;
		std::abort();
	}

	parse_command_line(argc, argv, lbfgs_param, program_param);
	load_data(program_param->featfile, insts, clslabels);

	sample_size = insts->rows();
	feat_size = insts->cols();
	Eigen::VectorXd x(4);
	x << 1, 2, 3, 4;

	std::cout << "Sample size : " << sample_size << std::endl;
	std::cout << "Feat   size : " << feat_size << std::endl;
	std::cout << "Non-zeros   : " << insts->nonZeros() << std::endl;
	
	lbfgsfloatval_t fx;
	lbfgsfloatval_t* w = lbfgs_malloc(feat_size);
	memset(w, 0, sizeof(lbfgsfloatval_t)*feat_size);
	program_param->featsize = feat_size;
	if(w == NULL){
		std::cerr << "allocation parameter array failed" << std::endl;
		std::exit(-1);
	}

	// initialize the parameter 
	Eigen::Map<Eigen::VectorXd> mappedW(w, feat_size, 1);
	mappedW = Eigen::MatrixXd::Constant(feat_size, 1, 0);

	// setup parameters
	lbfgs_param->m = 10;

	// train model for each class
	std::hash_set<int> clsids;
	for(int i=0; i<clslabels->rows(); ++i){
		if( clsids.count( clslabels->coeff(i) ) > 0 ){
			continue;
		}
		clsids.insert(clslabels->coeff(i));
	}

	for(std::hash_set<int>::iterator iter = clsids.begin();
		iter != clsids.end();
		++iter){
			std::cout << "train for label " << *iter << std::endl;
			train(&fx, w, feat_size, *iter,
				insts, clslabels, lbfgs_param, program_param);
			save_model(w, feat_size, program_param->outmodel, *iter);
	}
	return 0;
}