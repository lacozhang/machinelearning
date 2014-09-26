#include <iostream>
#include <fstream>
#include <ctime>
#include <boost/lexical_cast.hpp>
#include "train.h"
#include "util.h"

static lbfgsfloatval_t evaluate(
	void *instance,
	const lbfgsfloatval_t *x,
	lbfgsfloatval_t *g,
	const int n,
	const lbfgsfloatval_t step
	){
		double fx = 0;
		timeutil t;

		optimization_instance_t* optInst = (optimization_instance_t*)instance;
		Eigen::Map<Eigen::VectorXd> param((double*)x, 
			n, 1);
		Eigen::Map<Eigen::VectorXd> grad((double*)g,
			n, 1);

		// price for sparse A*x
		t.tic();
		Eigen::VectorXd xw = (*(optInst->samples_))*param;

		if( (xw.cols() != 1) || (xw.rows() != optInst->samples_->rows())){
			std::cerr << "Error, column size must be 1" << std::endl;
			std::cerr << "product of Xw is " << xw.rows() 
				<< " by " << xw.cols() << std::endl;
			std::exit(-1);
		}

		int effective_cnt = 0;
		Eigen::ArrayXd xwy(xw.rows());

		xwy.setZero();
		for(int i=0; i<xw.rows(); ++i){
			double z = xw.coeff(i)*(optInst->labels_->coeff(i));
			xwy.coeffRef(i) = z;
			if( z < 1 ){
				fx += (1-z)*(1-z);
				++effective_cnt;
			}
		}

		// fx /= optInst->labels_->rows();
		// l2 regularization value
		for(int i = optInst->prog_params_->l2start;
			i <= optInst->prog_params_->l2end;
			++i){
				fx += 0.5*optInst->prog_params_->l2c * std::pow(param.coeff(i), 2);
		}

		grad.setZero();

		for (int k = 0; k < optInst->samples_->outerSize(); ++k){

			double z = xwy.coeff(k);
			if(z >= 1){
				continue;
			}
			double factor = -2 * optInst->labels_->coeff(k) * (1-z);
			for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it( *(optInst->samples_) ,k); 
				it; ++it){
					int colIdx = it.col();
					grad.coeffRef(colIdx) += factor*it.value();
			}
		}

		// grad /= optInst->samples_->rows();

		//l2 regularization
		for(int i = optInst->prog_params_->l2start;
			i <= optInst->prog_params_->l2end;
			++i){
				grad.coeffRef(i) += optInst->prog_params_->l2c * param.coeff(i);
		}

		std::cout << "One computation of function value & gradient costs " << t.toc() << " seconds"
			<< std::endl;
		return fx;
}

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
	){

		printf("** Iteration %d:\n", k);
		printf("     fx = %f\n", fx);
		printf("     xnorm = %f, gnorm = %f, step size = %f\n", xnorm, gnorm, step);
		printf("\n");

		// check the accuracy
		optimization_instance_t* optInst = (optimization_instance_t*)instance;
		Eigen::Map<Eigen::VectorXd> param((double*)x, 
			n, 1);
		Eigen::Map<Eigen::VectorXd> grad((double*)g,
			n, 1);


		std::cout << "Training Accuracy " << test(x, n, 
			optInst->samples_, optInst->labels_) * 100
			<< std::endl;
		return 0;
}


double test(const lbfgsfloatval_t* parameter,
			int featsize,
			boost::shared_ptr<Eigen::SparseMatrix<double, Eigen::RowMajor> >& samples,
			boost::shared_ptr<Eigen::VectorXi>& labels){
				Eigen::Map<Eigen::VectorXd> param((double*)parameter, featsize, 1);
				Eigen::VectorXd xw = (*(samples))*param;

				double allcnt=0, rightcnt  =0 ;
				for(int i=0; i < labels->rows(); ++i){
					if( xw.coeff(i)*labels->coeff(i) > 0){
						rightcnt += 1;
					}
					allcnt += 1;
				}
				return rightcnt /allcnt;
}

void train(lbfgsfloatval_t* fx, lbfgsfloatval_t* parameter,
		   int featsize, int poslabel,
		   boost::shared_ptr<Eigen::SparseMatrix<double, Eigen::RowMajor> >& samples,
		   boost::shared_ptr<Eigen::VectorXi>& labels,
		   boost::shared_ptr<lbfgs_parameter_t>& opt_params,
		   boost::shared_ptr<derivative_parameter_t>& prog_params){

			   boost::shared_ptr<Eigen::VectorXi> instlabels( new Eigen::VectorXi(labels->rows()) );
			   instlabels->setOnes();
			   Eigen::Map<Eigen::VectorXd> map_param(parameter, featsize, 1);
			   for(int i=0; i < instlabels->rows(); ++i){
				   if( labels->coeff(i) != poslabel ){
					   instlabels->coeffRef(i) = -1;
				   }
			   }
			   map_param.setZero();
			   int ret=-1;
			   optimization_instance_t optInst(poslabel, samples, instlabels, prog_params);
			   do {
				   ret = lbfgs(featsize, parameter, fx,
					   evaluate, progress, (void*)&optInst, 
					   opt_params.get());

				   if( 0 == ret || LBFGSERR_MAXIMUMLINESEARCH == ret){
					   std::cout << "L-BFGS optimize successful" << std::endl;
				   }
				   std::cout << "L-BFGS optimization terminated with status code = " << ret << std::endl;
				   std::cout << "objective function value = " << *fx << std::endl;

				   std::cout << "Training Accuracy " << test(parameter, featsize, samples, instlabels) * 100
					   << std::endl;

			   } while( 0 );
}

void save_model(lbfgsfloatval_t* parameter, int featsize, std::string prefix, int poslabel){
	std::string path = prefix + "/" + boost::lexical_cast<std::string>(poslabel) + ".model";
	std::cout << "save model for " << poslabel
		<< std::endl;
	std::ofstream sink(path.c_str());
	if(! sink.is_open() ){
		std::cerr << "Error, create " << path << " failed"
			<< std::endl;
		return;
	}

	sink << featsize << std::endl;
	for(int i=0; i<featsize; ++i){
		sink << parameter[i]
		<< std::endl;
	}
	sink.close();
}