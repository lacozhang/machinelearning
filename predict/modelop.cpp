#include <iostream>
#include <boost/filesystem.hpp>
#include <boost/regex.hpp>
#include <boost/lexical_cast.hpp>
#include "modelop.h"

namespace {
	boost::regex model_name("(\\d+)\\.model");
}

Models::Models(std::string& modelpath){
	modelpath_ = modelpath;
	init();
}

void Models::init(){

	boost::filesystem::path p(modelpath_);
	
	if(! boost::filesystem::is_directory(p) ){
		std::cerr << p.filename() << " not directory" << std::endl;
		std::abort();
	}

	boost::filesystem::directory_iterator iter(p);
	boost::filesystem::directory_iterator end;

	for(; iter != end; ++iter){

		if( boost::filesystem::is_directory(*iter) ){
			std::cout << "dir" << std::endl;
			continue;
		}

		boost::smatch clsmatch;
		std::string smodelid = iter->path().string();
		std::cout << smodelid << std::endl;
		if(!boost::regex_search(smodelid, clsmatch, model_name)){
			std::cerr << "Weird, model does not match the name"
				<< std::endl;
			continue;
		}

		boost::shared_ptr<model> smodelptr;
		load_model(smodelid, smodelptr);

		smodelid.assign(clsmatch[1].first, clsmatch[1].second);
		std::cout << boost::lexical_cast<int>(smodelid) << std::endl;

		models_.insert( std::pair<int, boost::shared_ptr<model>>(
			boost::lexical_cast<int>(smodelid),
			smodelptr) );
	}
}

int Models::predict(sample& s){
	int cls = -1;
	double max_val = -10000;
	for( std::map<int, boost::shared_ptr<model> > ::iterator iter = models_.begin();
		iter != models_.end();
		++iter){
			if( iter->second->rows() < s.rows() ){
				std::cerr << " model size " << iter->second->rows() << "\t";
				std::cerr << " data  size " << s.cols() << "\t";
				std::cerr << "not equal" << std::endl;
				std::abort();
			}
			double p = s.dot( *(iter->second) );
			p = 1 / (1 + std::exp(-p) );

			if( p > max_val ){
				max_val = p;
				cls = iter->first;
			}
	}
	return cls;
}

void Models::load_model(std::string modelfile, boost::shared_ptr<model>& m){

	std::cout << "loading model " << modelfile << std::endl;
	std::ifstream src(modelfile);
	std::string buf;
	int featsize;
	
	if(! src.is_open() ){
		std::cerr << "error, can't open " << modelfile
			<< std::endl;
		std::abort();
	}

	try {
	std::getline(src, buf);
	} catch( std::exception& e){
		std::cerr << e.what() << std::endl;
		std::exit(-1);
	}

	try {
		featsize = boost::lexical_cast<int>(buf);
	} catch( std::exception& e){
		std::cout << e.what()
			<< std::endl;
		std::abort();
	}

	std::cout << "feat size : " << featsize << std::endl;
	m.reset( new model(featsize) );
	m->setZero();

	if( ! m.get() ){
		std::cerr << "Model allocation failed" << std::endl;
		std::abort();
	}
	int i =0;
	for(i=0; i<featsize && src.good(); ++i){
		std::getline(src, buf);
		try{
			m->coeffRef(i) = boost::lexical_cast<double>(buf);
		} catch(std::exception& e){
			std::cerr << e.what() << std::endl;
			std::cerr << " line " << i << std::endl;
			std::cerr << buf << std::endl;
			std::abort();
		}
	}

	if( i < featsize ){
		std::cerr << "error, load feature to only " << i
			<< std::endl;
		std::abort();
	}
}

double Models::zeroTest(int clsid, sample& s){
	return s.dot(*(models_[clsid]));
}