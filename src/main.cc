#include <iostream>
#include <opencv2/core/core.hpp>
#include <ceres/ceres.h>
#include <chrono>
//#include <vector>

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
    // initalize
    // curve parameters
    double m = 0.0;
    double c = 0.0;

    // max number of values
    int N = 100;

    // generate observations, x and y values
    cv::RNG rng;
    double sigma = 0.2;
    std::vector<double> x_vals, y_vals;

    for(int i=0; i<N; ++i) {
        // generate x and y values
        double x = i;
        x_vals.push_back(x);
        y_vals.push_back(exp(m * x + c) + rng.gaussian(sigma));
    }

    // the problem
    // iterate through the observations
    ceres::Problem problem;
    for(int j=0; j<N; ++j) {

        problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<Residual, 1, 1, 1> (
                    new Residual(x_vals[j], y_vals[j])), NULL, &m, &c);
    }

    // solver options
    ceres::Solver::Options options;
    options.max_num_iterations = 25;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;

    // solver summary
    ceres::Solver::Summary summary;
    Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << "\n";
    std::cout << "Initial m: " << 0.0 << " c: " << 0.0 << "\n";
    std::cout<< "Final m: " << m << " c: " << c << "\n";
    return 0;
}
