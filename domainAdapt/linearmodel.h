
#ifndef __LINEAR_MODEL_H__
#define __LINEAR_MODEL_H__

// Linear model with existing parameters
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <boost/shared_ptr.hpp>
#include <liblinearmodel.h>

class LinearModel {
public:
	typedef Eigen::SparseMatrix<double, Eigen::RowMajor> DataType;
	typedef Eigen::VectorXi LabelType;

	LinearModel(LiblinearModel& w, double shrink, double alpha, double lambda);

	double derivative(Eigen::Map<const Eigen::VectorXd>& param, 
		Eigen::Map<Eigen::VectorXd>& gradient);

	void score(Eigen::Map<const Eigen::VectorXd>& param,
		boost::shared_ptr< Eigen::SparseMatrix<double, Eigen::RowMajor> >& dat,
		boost::shared_ptr< Eigen::VectorXi > labels,
		double& precision, double& recall);

	boost::shared_ptr<DataType> & GetData();
	boost::shared_ptr<LabelType> & GetLabel();

	const boost::shared_ptr<DataType> & GetConstantData() const;
	const boost::shared_ptr<LabelType> & GetConstantLabel() const;
	
private:
	LiblinearModel& m_oldParam;
	boost::shared_ptr<Eigen::SparseMatrix<double, Eigen::RowMajor> > m_dat;
	boost::shared_ptr<Eigen::VectorXi> m_label;
	boost::shared_ptr<Eigen::VectorXd> m_parameter;

	double m_shrink;
	double m_alpha;
	double m_lambda;
};

#endif // __LINEAR_MODEL_H__