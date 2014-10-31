#include "parameter.h"

CmdParameters::CmdParameters(){
	m_shrink = m_alpha = m_lambda = 0.5;

	m_trainFile = m_existModel = m_output = "";

	m_listTestFiles.clear();
}

void parsecmd(int argc, char* argv[], CmdParameters& param){

	namespace po = boost::program_options;
	po::options_description desc;
	desc.add_options()
		("h,help", "produce help message")
		("train", po::value<std::string>(), "training file in liblinear forat")
		("test", po::value<std::vector<std::string> >()->multitoken(), "testing file in liblinear format")
		("model", po::value<std::string>(), "existing liblinear model")
		("shrink", po::value<double>(), "shrink parameters")
		("alpha", po::value<double>(), "l2 regularization parameters")
		("lambda", po::value<double>(), "l1 regularization parameters")
		("output", po::value<std::string>(), "model output");

	po::variables_map vm;

	try {
		po::store(po::parse_command_line(argc, argv, desc), vm);
	} catch(std::exception& e){
		std::cerr << "Error " << e.what() << std::endl;
		std::abort();
	}

	try{
		po::notify(vm);
	}catch(std::exception& e){
		std::cout << e.what() << std::endl;
		return ;
	}

	if( vm.count("h") || vm.count("help") ){
		std::cout << desc << std::endl;
		std::exit(-1);
	}

	std::cout << "Options Summary " << std::endl;
	if( vm.count("train") ){
		param.m_trainFile = vm["train"].as<std::string>();
		std::cout << "Training Data : " << param.m_trainFile << std::endl;
	}

	if(vm.count("test")){
		param.m_listTestFiles = vm["test"].as<std::vector<std::string> >();

		for(int i=0; i < param.m_listTestFiles.size(); ++i){
			std::cout << "Testing  Data : " << param.m_listTestFiles[i] << std::endl;
		}
	}

	if( vm.count("model")){
		param.m_existModel = vm["model"].as<std::string>();
		std::cout << "Old Model     : " << param.m_existModel << std::endl;
	}

	if(vm.count("output")){
		param.m_output = vm["output"].as<std::string>();
		std::cout << "Output Model  : " << param.m_output << std::endl;
	}

	if(vm.count("shrink")){
		param.m_shrink = vm["shrink"].as<double>();
		std::cout << "Shrink Param  : " << param.m_shrink << std::endl;
	}

	if(vm.count("alpha")){
		param.m_alpha = vm["alpha"].as<double>();
		std::cout << "Alpha Param   : " << param.m_alpha << std::endl;
	}

	if(vm.count("lambda")){
		param.m_lambda = vm["lambda"].as<double>();
		std::cout << "Lambda Param  : " << param.m_lambda << std::endl;
	}
}