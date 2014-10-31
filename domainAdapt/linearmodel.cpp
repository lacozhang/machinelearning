#include <iostream>
#include <cmath>
#include "linearmodel.h"
#include "../utils/util.h"

LinearModel::LinearModel(LiblinearModel& w, double shrink, double alpha, double lambda): m_oldParam(w) {
	m_shrink = shrink;
	m_alpha = alpha;
	m_lambda = lambda;
	m_dat.reset( new DataType() );
	m_label.reset( new LabelType() );
	m_parameter.reset( new Eigen::VectorXd() );
}

boost::shared_ptr<LinearModel::DataType> & LinearModel::GetData(){
	return m_dat;
}

boost::shared_ptr<LinearModel::LabelType> & LinearModel::GetLabel(){
	return m_label;
}

const boost::shared_ptr<LinearModel::DataType> & LinearModel::GetConstantData() const {
	return m_dat;
}

const boost::shared_ptr<LinearModel::LabelType> & LinearModel::GetConstantLabel() const {
	return m_label;
}

// This function used to calculate the gradient of linear models
double LinearModel::derivative(Eigen::Map<const Eigen::VectorXd>& param, Eigen::Map<Eigen::VectorXd>& gradient){

	// Use squared hinge loss
	Eigen::VectorXd Xw = (*m_dat) * param;
	// std::cout << "Xw costs" << t.toc() << " seconds" << std::endl;
	Eigen::VectorXd yXw( Xw.rows() );

	// calculate the function value
	double funcval = 0;
	for(int i=0; i<m_dat->rows(); ++i){

		yXw.coeffRef(i) = m_label->coeff(i) * Xw.coeff(i);
		funcval += std::pow(std::max(0.0, 1 - yXw.coeff(i)), 2);

	}

	funcval /= m_dat->rows();
	funcval *= (1 - m_alpha);
	// std::cout << "Func val costs " << t.toc() << " seconds" << std::endl;

	// calculate the shrink parameter value
	Eigen::VectorXd paramdiff = param - m_oldParam.getparam().toDense();
	funcval += 0.5*m_shrink* std::pow(paramdiff.lpNorm<2>(), 2);
	// std::cout << "Shrink value " << t.toc() << " seconds" << std::endl;

	// calculate the l2 regularization 
	funcval += m_lambda * (1 - m_alpha)* std::pow(param.lpNorm<2>(), 2) * 0.5;

	// calculate the l1 regularization
	funcval += m_lambda * m_alpha * param.lpNorm<1>();
	// std::cout << "L1 & L2 reg value costs " << t.toc() << " seconds" << std::endl;

	gradient.setZero();

	for(int i=0; i < m_dat->outerSize(); ++i){
		double z = yXw.coeff(i);

		if( z > 1 ){
			continue;
		}

		double factor = -2 * ( 1 - z ) * m_label->coeff(i);

		if( m_label->coeff(i) < 0 && Xw.coeff(i) > 0 ){
			factor *= 2.0;
		}

		Eigen::VectorXd singleGradient( param.rows() );
		for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it( *(m_dat) ,i); 
				it; ++it){
					int colIdx = it.col();
					gradient.coeffRef(colIdx) += factor*it.value();
		}
	}

	// std::cout << "Gradient costs " << t.toc() << std::endl;

	gradient /= yXw.rows();
	gradient *= (1 - m_alpha);

	gradient += m_shrink * paramdiff;
	gradient += m_lambda * (1 - m_alpha) * param * 0.5;

	/*
	std::cout <<"Func value       : " << funcval << std::endl;
	std::cout <<"Norm of gradient : " << gradient.lpNorm<2>() << std::endl;
	std::cout <<"Norm of weight   : " << param.lpNorm<2>() << std::endl;
	*/

	return funcval;
}

void LinearModel::score(Eigen::Map<const Eigen::VectorXd>& param,
		boost::shared_ptr< Eigen::SparseMatrix<double, Eigen::RowMajor> >& dat,
		boost::shared_ptr< Eigen::VectorXi > labels,
		double& precision, double& recall){
			*m_parameter = param;

			PRUtil::score(*m_parameter, *dat, *labels, precision, recall);
}