
// loading model from liblinear

#ifndef __LIB_LINEAR_MODEL_H__
#define __LIB_LINEAR_MODEL_H__

#include <vector>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <boost/shared_ptr.hpp>


struct LiblinearModel {

	LiblinearModel(){
		m_nrClasses = 0;
		m_parameters.reset();
	}

	LiblinearModel(std::string filename);

	double coeff(int i) const;

	void TestData(boost::shared_ptr<Eigen::SparseMatrix<double, Eigen::RowMajor> > & testData,
		boost::shared_ptr<Eigen::VectorXi>& labels,
		double& precision, double& recall);

	bool saveModel(std::string filename);
	Eigen::SparseVector<double>& getparam() const;

private:
	void parseModel(std::string filename);
	void ReadWeights(std::ifstream& src, int featcnt, Eigen::VectorXd& param);

	int m_nrClasses;
	int m_featCnt;
	std::vector<std::string> m_headers;
	std::vector<int> m_labels;
	boost::shared_ptr< Eigen::SparseVector<double> > m_parameters;
};

#endif // __LIB_LINEAR_MODEL_H__