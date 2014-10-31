
#include <lbfgs.h>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <boost/shared_ptr.hpp>
#include "parameters.h"

#ifndef __TRAIN_H__
#define __TRAIN_H__

void train(lbfgsfloatval_t* fx, 
		   lbfgsfloatval_t* parameter,
		   int featsize, 
		   int poslabel,
		   boost::shared_ptr<Eigen::SparseMatrix<double, Eigen::RowMajor> >& samples,
		   boost::shared_ptr<Eigen::VectorXi>& labels,
		   boost::shared_ptr<lbfgs_parameter_t>& params,
		   boost::shared_ptr<derivative_parameter_t>& prog_params);

void save_model(lbfgsfloatval_t* parameter, int feat_size, std::string filename, int poslabel);

double test(const lbfgsfloatval_t* parameter,
			int featsize,
			boost::shared_ptr<Eigen::SparseMatrix<double, Eigen::RowMajor> >& samples,
			boost::shared_ptr<Eigen::VectorXi>& labels);

static lbfgsfloatval_t evaluate(
    void *instance,
    const lbfgsfloatval_t *x,
    lbfgsfloatval_t *g,
    const int n,
    const lbfgsfloatval_t step
    );

static int progress(
    void *instance,
    const lbfgsfloatval_t *x,
    const lbfgsfloatval_t *g,
    const lbfgsfloatval_t fx,
    const lbfgsfloatval_t xnorm,
    const lbfgsfloatval_t gnorm,
    const lbfgsfloatval_t step,
    int n,
    int k,
    int ls
    );

#endif