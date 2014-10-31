#include <iostream>
#include <boost/shared_ptr.hpp>
#include <boost/shared_array.hpp>
#include <liblinearmodel.h>
#include "linearmodel.h"
#include <dataop.h>
#include "../include/lbfgs.h"
#include "../utils/util.h"
#include "parameter.h"

// with options -m existing_model --train train_file(liblinear format) --test test data

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

void train(lbfgsfloatval_t* fx, lbfgsfloatval_t* parameter,
		   int featsize, boost::shared_ptr<lbfgs_parameter_t>& opt_params,
		   double shrink, double l2reg, double l1reg,
		   std::vector<TestPair>* model);

boost::shared_ptr<lbfgs_parameter_t> bfgsParam;
boost::shared_array<lbfgsfloatval_t> weight;
boost::shared_ptr<LinearModel> targetModel;
boost::shared_ptr<LiblinearModel> existedModel;
boost::shared_ptr< std::vector<TestPair> > testingFiles;

int main(int argc, char* argv[]){

	CmdParameters param;
	parsecmd(argc, argv, param);

	existedModel.reset( new LiblinearModel(param.m_existModel) );
	targetModel.reset( new LinearModel((*existedModel), param.m_shrink, param.m_alpha, param.m_lambda ) );

	// updating testing files
	testingFiles.reset( new std::vector<TestPair>() );

	int featsize = existedModel->getparam().rows();
	std::cout << "Feat size " << featsize << std::endl;
	load_data(param.m_trainFile, targetModel->GetData(), targetModel->GetLabel(), -1, featsize);
	std::cout << "Load Train Data Finished" << std::endl;
	
	bfgsParam.reset( new lbfgs_parameter_t() );
	if(! bfgsParam.get()){
		std::cerr << "new for lbfgs parameter failed"
			<< std::endl;
		std::exit(-1);
	}

	lbfgs_parameter_init( bfgsParam.get() );
	bfgsParam->orthantwise_c = param.m_lambda * param.m_alpha;
	bfgsParam->linesearch = LBFGS_LINESEARCH_BACKTRACKING;

	// loading test data
	std::cout << "loading test data" << std::endl;
	for(int i=0; i < param.m_listTestFiles.size(); ++i){
		std::cout << " *** " << std::endl;
		std::cout << "load set " << param.m_listTestFiles[i] << std::endl;
		std::cout << " *** " << std::endl;
		TestPair tmp;
		boost::shared_ptr<Eigen::SparseMatrix<double, Eigen::RowMajor> > sample;
		boost::shared_ptr<Eigen::VectorXi> labels;

		sample.reset( new Eigen::SparseMatrix<double, Eigen::RowMajor>() );
		labels.reset( new Eigen::VectorXi() );

		load_data( param.m_listTestFiles[i], sample, labels, -1, featsize );

		tmp.samples = sample;
		tmp.labels = labels;
		tmp.setname = param.m_listTestFiles[i];

		testingFiles->push_back( tmp );
	}

	weight.reset( new lbfgsfloatval_t[featsize] );
	if( ! weight.get() ){
		std::cerr << "new for weight failed"
			<< std::endl;
		std::exit(-1);
	}
	memset( weight.get(), 0.1, featsize*sizeof(lbfgsfloatval_t));

	lbfgsfloatval_t funcval;

	train(&funcval, weight.get(), 
		featsize, bfgsParam, 
		param.m_shrink, param.m_alpha, param.m_lambda,
		testingFiles.get());

	std::cout << "Evaluating Original P/R" << std::endl;
	double p,r;
	for(int i=0; i < testingFiles->size(); ++i){
		existedModel->TestData( (*testingFiles)[i].samples,
			(*testingFiles)[i].labels, p, r);
		std::cout << "Set Name " << (*testingFiles)[i].setname
			<< std::endl;
		std::cout << "P: " << p << " R: " << r << std::endl;
	}

	return 0;
}

static lbfgsfloatval_t evaluate(
	void *instance,
	const lbfgsfloatval_t *x,
	lbfgsfloatval_t *g,
	const int n,
	const lbfgsfloatval_t step
	){
		Eigen::Map<const Eigen::VectorXd > param(x, n, 1);
		Eigen::Map<Eigen::VectorXd > grad(g, n, 1);

		double funcval = targetModel->derivative( param, grad );

		return funcval;
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

		std::cout << "** Iteration " << k << std::endl;
		std::cout << "     fx = " << fx << std::endl;
		std::cout << "     xnorm = " << xnorm 
			<< ", gnorm = " << gnorm 
			<< ", step size = " << step << std::endl;

		// check the accuracy
		Eigen::Map<const Eigen::VectorXd> param((double*)x, 
			n, 1);
		Eigen::Map<const Eigen::VectorXd> grad((double*)g,
			n, 1);

		std::cout << "Model performance :" << std::endl;
		std::vector<TestPair>& testData( *(static_cast<std::vector<TestPair>*>(instance)) );

		for(std::vector<TestPair>::iterator iter = testData.begin() ; iter != testData.end(); ++iter){
			std::cout << "Test set name : " << iter->setname << std::endl;
			double p, r;
			targetModel->score( param, iter->samples, iter->labels, p, r);
			std::cout << "Precision     : " << p << "\tRecall : " << r << std::endl;
		}


		if( std::abs(step) < 1e-10 ){
			return 1;
		}

		return 0;
}

void train(lbfgsfloatval_t* fx, lbfgsfloatval_t* parameter,
		   int featsize, boost::shared_ptr<lbfgs_parameter_t>& opt_params,
		   double shrink, double l2reg, double l1reg,
		   std::vector<TestPair>* model){

			   std::cout << "Start Trainig .. " << std::endl;
			   int ret=lbfgs(featsize, parameter, fx, evaluate, progress, (void*)model, bfgsParam.get());
			   std::cout << "ret = " << LbfgsRet::Error(ret) << std::endl;
}