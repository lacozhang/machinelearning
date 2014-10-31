#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include "liblinearmodel.h"
#include "../utils/util.h"

LiblinearModel::LiblinearModel(std::string filename){
	parseModel(filename);
}

void LiblinearModel::ReadWeights(std::ifstream& src, int featcnt, Eigen::VectorXd& param){

	// read in all the weights
	param.resize( featcnt );
	param.fill(0);

	double featval;
	int read_cnt = 0;
	for(; read_cnt < m_featCnt && src.good(); ++read_cnt){
		src >> featval;
		param.coeffRef(read_cnt) = featval;
	}

	if( src.eof() ){
		std::cout << "Read to End Of File" << std::endl;
	}

	if( read_cnt < m_featCnt ){
		std::cerr << "Something error in reading weights"
			<< std::endl;
		std::cerr << "Read count " << read_cnt << std::endl;
		std::cerr << "Feat Count " << featcnt << std::endl;
		std::exit(-1);
	}

	std::cout << "Read count " << read_cnt << std::endl;
	std::cout << "Feat Count " << featcnt << std::endl;
}

double LiblinearModel::coeff(int index) const {
	return m_parameters->coeff(index);
}

Eigen::SparseVector<double>& LiblinearModel::getparam() const {
	return * m_parameters;
}

// read models from liblinear file
void LiblinearModel::parseModel(std::string src){
	std::ifstream ifsrc( src );
	std::string buff;

	if( !ifsrc.is_open() ){
		std::cerr << "Error, open " << src << " failed";
		std::exit(-1);
	}

	// read in solver type
	ifsrc >> buff;
	m_headers.push_back( buff );
	ifsrc >> buff;
	m_headers.push_back( buff );

	// read #classes
	ifsrc >> buff;
	m_headers.push_back( buff );

	ifsrc >> m_nrClasses;
	std::cout << "#classes" << "\t" ;
	std::cout << m_nrClasses << std::endl;

	// read labels for each classes || labels 1 2
	ifsrc >> buff;
	m_headers.push_back(buff);

	int labelbuff;
	for(int i=0; i< m_nrClasses; ++i){
		ifsrc >> labelbuff;
		m_labels.push_back(labelbuff);
	}

	if( m_labels.size() != 2){
		std::cerr << "Error, only support 2-class classification model"
			<< std::endl;
		std::cerr << "Label size " << m_labels.size()
			<< std::endl;
		std::exit(-1);
	}

	// read #features || nr_features 02343
	ifsrc >> buff;
	m_headers.push_back( buff );

	ifsrc >> m_featCnt;
	std::cout << "Feature count " << "\t";
	std::cout << m_featCnt << std::endl;

	// read bias term value || bias 0
	ifsrc >> buff;
	m_headers.push_back(buff);

	ifsrc >> buff;
	m_headers.push_back(buff);

	// read parameters w
	ifsrc >> buff;
	m_headers.push_back(buff);

#ifdef DEBUG
	std::cout << "output headers" << std::endl;
	for(int i=0; i<m_headers.size(); ++i){
		std::cout << m_headers[i] << std::endl;
	}
#endif

	int n_w = m_nrClasses > 2 ? m_nrClasses : 1;
	
	if( n_w > 1 ){
		std::cerr << "Current only supoort for 2-class" << std::endl;
	}

	// estimate the nnz
	Eigen::VectorXd tmp_vec;
	ReadWeights(ifsrc, m_featCnt, tmp_vec);

	// Construct sparse weight object
	m_parameters.reset( new Eigen::SparseVector<double>( m_featCnt ) );
	if(! m_parameters.get() ){
		std::cerr << "Error, File allocation failed" << std::endl;
	}

	int nnz = 0;
	for(int i=0; i<tmp_vec.rows(); ++i){
		if( std::abs( tmp_vec[i] ) > 1e-21 ){
			nnz += 1;
		}
	}

	m_parameters->reserve( nnz + 20 );
	for(int i=0; i<tmp_vec.rows(); ++i){
		if( std::abs( tmp_vec[i] ) > 1e-21 ){
			m_parameters->insert(i) = tmp_vec[i];
		}
	}

#ifdef DEBUG
	for(Eigen::SparseVector<double>::InnerIterator iter(*m_parameters); iter; ++iter){
		std::cout << "idx " << iter.index() << std::endl;
		std::cout << "val " << iter.value() << std::endl;
	}
#endif DEBUG
}


void LiblinearModel::TestData(boost::shared_ptr<Eigen::SparseMatrix<double, Eigen::RowMajor> > & testData, 
							  boost::shared_ptr<Eigen::VectorXi>& labels,
							  double& precision, double& recall){
	Eigen::VectorXd dw = m_parameters->toDense();
	PRUtil::score(dw, *testData, *labels, precision, recall);
}