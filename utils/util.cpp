#include "util.h"

timeutil::timeutil(){
}

void timeutil::tic(){
	t_ = clock();
}

double timeutil::toc(){
	return ((double)t_)/CLOCKS_PER_SEC;
}


std::string LbfgsRet::Error(int errorcode){
	std::string ret;
	switch(errorcode){
	case 0:
		ret = "Optimizaiton Successful";
		break;
	case 1:
		ret = "LBFGS_STOP";
		break;
	case 2:
		ret = "LBFGS_ALREADY_MINIMIZED";
		break;
	case -1024:
		ret = "LBFGSERR_UNKNOWNERROR";
		break;
	case -1023:
		ret = "LBFGSERR_LOGICERROR";
		break;
    /** Insufficient memory. */
	case -1022:
		ret = "LBFGSERR_OUTOFMEMORY";
		break;
    /** The minimization process has been canceled. */
	case -1021:
		ret = "LBFGSERR_CANCELED";
		break;
    /** Invalid number of variables specified. */
	case -1020:
		ret = "LBFGSERR_INVALID_N";
		break;
    /** Invalid number of variables (for SSE) specified. */
	case -1019:
		ret = "LBFGSERR_INVALID_N_SSE";
		break;
    /** The array x must be aligned to 16 (for SSE). */
	case -1018:
		ret = "LBFGSERR_INVALID_X_SSE";
		break;
    /** Invalid parameter lbfgs_parameter_t::epsilon specified. */
	case -1017:
		ret = "LBFGSERR_INVALID_EPSILON";
		break;
    /** Invalid parameter lbfgs_parameter_t::past specified. */
	case -1016:
		ret = "LBFGSERR_INVALID_TESTPERIOD";
		break;
    /** Invalid parameter lbfgs_parameter_t::delta specified. */
	case -1015:
		ret = "LBFGSERR_INVALID_DELTA";
		break;
    /** Invalid parameter lbfgs_parameter_t::linesearch specified. */
	case -1014:
		ret = "LBFGSERR_INVALID_LINESEARCH";
		break;
    /** Invalid parameter lbfgs_parameter_t::max_step specified. */
	case -1013:
		ret = "LBFGSERR_INVALID_MINSTEP";
		break;
    /** Invalid parameter lbfgs_parameter_t::max_step specified. */
	case -1012:
		ret = "LBFGSERR_INVALID_MAXSTEP";
		break;
    /** Invalid parameter lbfgs_parameter_t::ftol specified. */
	case -1011:
		ret = "LBFGSERR_INVALID_FTOL";
		break;
    /** Invalid parameter lbfgs_parameter_t::wolfe specified. */
	case -1010:
		ret = "LBFGSERR_INVALID_WOLFE";
		break;
    /** Invalid parameter lbfgs_parameter_t::gtol specified. */
	case -1009:
		ret = "LBFGSERR_INVALID_GTOL";
		break;
    /** Invalid parameter lbfgs_parameter_t::xtol specified. */
	case -1008:
		ret = "LBFGSERR_INVALID_XTOL";
		break;
    /** Invalid parameter lbfgs_parameter_t::max_linesearch specified. */
	case -1007:
		ret = "LBFGSERR_INVALID_MAXLINESEARCH";
		break;
    /** Invalid parameter lbfgs_parameter_t::orthantwise_c specified. */
	case -1006:
		ret = "LBFGSERR_INVALID_ORTHANTWISE";
		break;
    /** Invalid parameter lbfgs_parameter_t::orthantwise_start specified. */
	case -1005:
		ret = "LBFGSERR_INVALID_ORTHANTWISE_START";
		break;
    /** Invalid parameter lbfgs_parameter_t::orthantwise_end specified. */
	case -1004:
		ret = "LBFGSERR_INVALID_ORTHANTWISE_END";
		break;
    /** The line-search step went out of the interval of uncertainty. */
	case -1003:
		ret = "LBFGSERR_OUTOFINTERVAL";
		break;
    /** A logic error occurred; alternatively, the interval of uncertainty
        became too small. */
	case -1002:
		ret = "LBFGSERR_INCORRECT_TMINMAX";
		break;
    /** A rounding error occurred; alternatively, no line-search step
        satisfies the sufficient decrease and curvature conditions. */
	case -1001:
		ret = "LBFGSERR_ROUNDING_ERROR";
		break;
    /** The line-search step became smaller than lbfgs_parameter_t::min_step. */
	case -1000:
		ret = "LBFGSERR_MINIMUMSTEP";
		break;
    /** The line-search step became larger than lbfgs_parameter_t::max_step. */
	case -999:
		ret = "LBFGSERR_MAXIMUMSTEP";
		break;
    /** The line-search routine reaches the maximum number of evaluations. */
	case -998:
		ret = "LBFGSERR_MAXIMUMLINESEARCH";
		break;
    /** The algorithm routine reaches the maximum number of iterations. */
	case -997:
		ret = "LBFGSERR_MAXIMUMITERATION";
    /** Relative width of the interval of uncertainty is at most
        lbfgs_parameter_t::xtol. */
	case -996:
		ret = "LBFGSERR_WIDTHTOOSMALL";
    /** A logic error (negative line-search step) occurred. */
	case -995:
		ret = "LBFGSERR_INVALIDPARAMETERS";
		break;
    /** The current search direction increases the objective function value. */
	case -994:
		ret = "LBFGSERR_INCREASEGRADIENT";
		break;
	default:
		ret = "Unknown Error";
	}

	return ret;
}

void PRUtil::score(Eigen::VectorXd& parameter,
		Eigen::SparseMatrix<double, Eigen::RowMajor>& data,
		Eigen::VectorXi& labels,
		double& precision, double& recall){
				precision = recall = 0;

	double tp = 0, fp = 0, tn = 0, fn = 0;

	Eigen::VectorXd Xw = data * parameter;

	if( Xw.rows() != labels.rows() ){
		std::cerr << "The number of samples doesn't equal the labels"
			<< std::endl;
		std::abort();
	}

	for(int i=0; i< labels.rows(); ++i){
		switch (labels.coeff(i))
		{
		case 1:
			{
				if( Xw.coeff(i) > 0 ){
					tp += 1;
				} else {
					fn += 1;
				}
			}
			break;
		case -1:
			{
				if( Xw.coeff(i) < 0){
					tn += 1;
				} else {
					fp += 1;
				}
			}
			break;
		default:
			std::cerr << "Error, unknow classes"
				<< std::endl;
			break;
		}
	}

	precision = tp / (tp + fp);
	recall = tp / (tp + fn);
}