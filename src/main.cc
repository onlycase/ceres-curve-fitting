#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <ceres/ceres.h>
#include <chrono>


// define a templated object to evaluate the residual for each obs
struct Residual {
    Residual(double x, double y) : x_(x), y_(y) { }

    template <typename T>
    bool operator()(const T* const m, const T* const c, T* residual) const {
        residual[0] = T(y_) - exp(m[0] * T(x_) + c[0]);
        return true;
    }

    private:
    // observations
    const double x_, y_;
};


int main(int argc, char* argv[]) {

    // number of observations
    int N = 100;
    // initial values
    double m = 0.0;
    double c = 0.0;
    // curve parameters for generating data
    double m_obs = 0.3;
    double c_obs = 0.1;

    cv::RNG rng;
    double sigma = 0.2;
    std::vector<double> x_obs, y_obs;

    for(int i=0; i<N; ++i) {
        // generate x and y values
        double x_i = i/20.0;
        x_obs.push_back(x_i);
        y_obs.push_back(exp(m_obs * x_i + c_obs) + rng.gaussian(sigma));
    }

    ceres::Problem problem;
    for(int j=0; j<N; ++j) {
        problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<Residual, 1, 1, 1> (
                    new Residual(x_obs[j], y_obs[j])), NULL, &m, &c);
    }

    ceres::Solver::Options options;
    options.max_num_iterations = 25;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;

    // solver summary
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << summary.FullReport() << std::endl;
    std::cout << "Initial m: " << 0.0 << " c: " << 0.0 << std::endl;
    std::cout<< "Final m: " << m << " c: " << c << std::endl;;



    // writes to a file in build directory
    std::ofstream outfile("ceres-output.txt");

    // generating ground truth values and writing
    for(int k=0; k<N; ++k) {
        double x_gt = k/20.0;
        outfile << x_gt << "," << exp(m_obs*x_gt+c_obs) << "," << y_obs[k] << "," << exp(m*x_gt+c) << "\n"; 
    }

    outfile.close();

    return 0;
}
