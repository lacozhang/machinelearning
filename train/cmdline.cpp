#include <iostream>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include "cmdline.h"

void parse_command_line(int argc, char* argv[], 
						boost::shared_ptr<lbfgs_parameter_t>& param,
						boost::shared_ptr<derivative_parameter_t>& derivative_param){

	bool bforget = false;
	namespace po = boost::program_options;
	po::options_description desc("Program options for train");
	desc.add_options()
		("help,h", "produce help message")
		("feat,f", po::value<std::string>()->required(), "feature files for training")
		("l1start", po::value<int>()->default_value(1), "start feature index for l1 norm")
		("l1end", po::value<int>()->default_value(-1), "end feature index for l1 norm")
		("l1c", po::value<double>()->default_value(1.0f), "default value for regularization parameter")
		("l2start", po::value<int>()->default_value(1), "start feature index for l2 norm")
		("l2end", po::value<int>()->default_value(-1), "end feature index for l2 norm")
		("l2c",  po::value<double>()->default_value(1.0f), "regularization parameter value for l2 norm")
		("iter", po::value<int>()->default_value(40), "number of iterations for optimization")
		("output", po::value<std::string>()->default_value("output.model"), "file store model");

	po::variables_map vm;

	try {
		po::store(po::parse_command_line(argc, argv, desc), vm);
	} catch(std::exception& e){
		std::cerr << e.what() << std::endl;
	}

	try {
		po::notify(vm);
	} catch(std::exception& e){
		std::cerr << e.what() << std::endl;
		bforget = true;
	}

	if( vm.count("help") ){
		std::cout << desc << std::endl;
		return;
	}

	if( bforget ){
		std::exit(-1);
	}
	
	int l1start = vm["l1start"].as<int>(), l1end = vm["l1end"].as<int>();
	int l2start = vm["l2start"].as<int>(), l2end = vm["l2end"].as<int>();
	double l1c = vm["l1c"].as<double>(), l2c = vm["l2c"].as<double>();
	int iter = vm["iter"].as<int>();
	std::string featfile = vm["feat"].as<std::string>(); 
	std::string	outmodel = vm["output"].as<std::string>();

	/* sanity check first */
	boost::filesystem::path p(featfile);
	if( !boost::filesystem::exists(p) ){
		
		std::cerr << "file " << featfile << " does not exist" << std::endl;
		std::abort();
	}
		
	if( !boost::filesystem::is_regular_file(p) ){
		std::cerr << "file " << p.filename() << " is not a regular file" << std::endl;
		std::abort();
	}

	boost::filesystem::path m(outmodel);
	if(!boost::filesystem::is_directory(m)){
		std::cerr << "output prefix must be a directory" << std::endl;
		std::abort();
	}

	if( (l1start >= l1end) || (l2start >= l2end) ){
		std::cerr << "end index must larget than start index" << std::endl;
		std::abort();
	}

	if( (l1c < 0 ) || (l2c < 0) ){
		std::cerr << "regularization parameter can not be negative" << std::endl;
		std::abort();
	}

	/* Start to print out the parameters */
	std::cout << "Feature range for L1 normalization : [" 
		<< l1start << ", " 
		<< l1end << "]"
		<< std::endl;

	std::cout << "Regularization parameter lambda_0  : " 
		<< l1c << std::endl;

	std::cout << "Feature range for L2 normalization : ["
		<< l2start << ", "
		<< l2end << "]"
		<< std::endl;

	std::cout << "Regularization parameter lambda_1  : "
		<< l2c << std::endl;

	std::cout << " **** Reading training data from   : " 
		<< featfile << std::endl;

	std::cout << " **** Train iterations             : "
		<< iter << std::endl;

	std::cout << " **** model will output to **      : "
		<< outmodel << std::endl;

	/* set up parameters for L-BFGS */
	lbfgs_parameter_init(param.get());
	param->linesearch = LBFGS_LINESEARCH_BACKTRACKING;
	if( abs(l1c) > 1e-3 ){
		param->orthantwise_c = l1c;
		param->orthantwise_start = l1start;
		param->orthantwise_end = l1end;
	}

	param->max_iterations = iter;

	/* set up parameters for function evaluation */
	derivative_param->l2start = l2start;
	derivative_param->l2end = l2end;
	derivative_param->l2c = l2c;
	derivative_param->featfile = featfile;
	derivative_param->outmodel = outmodel;
	derivative_param->l1start = l2start;
	derivative_param->l1end = l1end;
}