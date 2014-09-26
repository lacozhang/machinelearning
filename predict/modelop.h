
#include <boost/shared_ptr.hpp>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <map>

typedef Eigen::SparseVector<double> sample;
typedef Eigen::VectorXd model;

#ifndef __MODEL_OP_H__
#define __MODEL_OP_H__



class Models {
public:
	Models(std::string& modeldir);

	// predict the label according to current model
	int predict(sample& s);
	double zeroTest(int clsid, sample& s);
private:

	void init();
	void load_model(std::string modelfile, boost::shared_ptr<model>& m);
	std::map<int, boost::shared_ptr<model> > models_;

private:
	std::string modelpath_;
};

#endif // __MODEL_OP_H__