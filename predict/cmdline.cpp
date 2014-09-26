#include "cmdline.h"
#include <iostream>
#include <boost/program_options.hpp>

void parse_cmd(int argc, char* argv[], std::string& modelpath, std::string& input, std::string& output){

	namespace po = boost::program_options;
	po::options_description opts("program for predict classification result");
	opts.add_options()
		("h,help", "produce help message")
		("model", po::value<std::string>(), "directory contains each model")
		("input", po::value<std::string>(), "input feat file")
		("output", po::value<std::string>(), "predict results");

	po::variables_map vm;
	try {
		po::store(po::parse_command_line(argc, argv, opts), vm);
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
		std::cout << opts << std::endl;
		return;
	}

	input = vm["input"].as<std::string>();
	modelpath = vm["model"].as<std::string>();
	output = vm["output"].as<std::string>();

	std::cout << "***  input file : " << input 
		<< std::endl;
	std::cout << "***  model file : " << modelpath
		<< std::endl;
	std::cout << "*** output file : " << output 
		<< std::endl;
}