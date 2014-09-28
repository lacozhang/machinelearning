#include <iostream>
#include <fstream>
#include <boost/shared_ptr.hpp>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include "cmdline.h"
#include "modelop.h"
#include <dataop.h>

typedef Eigen::SparseMatrix<double, Eigen::RowMajor> DataMatrix;
typedef Eigen::VectorXi LabelVector;

boost::shared_ptr<DataMatrix> testData;
boost::shared_ptr<LabelVector> testLabels;

double savePredict(Models& m, boost::shared_ptr<DataMatrix>& dat,
				   boost::shared_ptr<LabelVector>& label,
				   std::string& out);

double usualTest(Models& m, sample& dat, int label);

bool bDebug = false;

int main(int argc, char* argv[]){

	std::string input, model, output;
	parse_cmd(argc, argv, model, input, output, bDebug);
	Models m(model);

	std::cout << " loading model " << std::endl;

	load_data(input, testData, testLabels);

	std::cout <<" rows " << testData->rows() << " cols " << testData->cols()
		<< std::endl;

	double accuracy = savePredict(m, testData, testLabels, output);

	std::cout << "Total Accuracy is " << accuracy << std::endl;
	return 0;
}

double savePredict(Models& m, boost::shared_ptr<DataMatrix>& dat, boost::shared_ptr<LabelVector>& labels, std::string& out){

	std::ofstream sink(out);
	if(!sink.is_open()){
		std::cerr << "Crate file " << out << " failed"
			<< std::endl;
	}

	double total = 0, right = 0;

	std::cout << "Total of " << testData->rows() << " queries" << std::endl;

	for(int i=0; i < testData->rows(); ++i){
		sample k = testData->row(i);
		int pred=m.predict(k);

		sink << pred << "\t" << labels->coeff(i) << std::endl;

		if( pred != testLabels->coeff(i) && bDebug ){
			usualTest(m, k, testLabels->coeff(i));
		}

		total++;
		if( pred == testLabels->coeff(i) ){
			right ++;
		}
	}
	return right/total;

}

double usualTest(Models& m, sample& s, int label){

	double accuracy = 0;

	double total=0, right =0;

	std::cout << "Error predicts" << std::endl;

	double a = 0;

	std::cout << "real label " << label
		<< std::endl;
	switch (label)
	{
	case 1:
		a = m.zeroTest(1, s);
		std::cout << "class 1 output " << a
			<< std::endl;
		a = m.zeroTest(2, s);
		std::cout << "class 2 output " << a
			<< std::endl;
		break;
	case 2:
		a = m.zeroTest(1, s);
		std::cout << "class 1 output " << a
			<< std::endl;
		a = m.zeroTest(2, s);
		std::cout << "class 2 output " << a
			<< std::endl;
		break;
	}

	return 0;
}