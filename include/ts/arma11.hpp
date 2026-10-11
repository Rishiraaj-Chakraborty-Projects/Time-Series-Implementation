#pragma once

#include <cstddef>
#include <random>
#include <vector>

namespace ts {

struct ARMA11Parameters {
    double phi{0.0};
    double theta{0.0};
    double intercept{0.0};
    double sigma{1.0};
};

struct ARMA11Fit {
    ARMA11Parameters parameters;
    double objective{};
    std::size_t observations{};
};

// Simulates X_t = c + phi X_{t-1} + eta_t + theta eta_{t-1}.
std::vector<double> simulate_arma11(const ARMA11Parameters& params,
                                     std::size_t n,
                                     unsigned int seed = 42,
                                     std::size_t burn_in = 500);

// Theoretical stationary mean and variance for |phi| < 1.
double theoretical_mean(const ARMA11Parameters& params);
double theoretical_variance(const ARMA11Parameters& params);

// Conditional sum-of-squared-innovations (CSS) fit over stationary (|phi| < 1)
// and invertible (|theta| < 1) parameter space. Profiles out intercept via sample mean.
ARMA11Fit fit_arma11_grid(const std::vector<double>& x,
                          double phi_min = -0.95,
                          double phi_max = 0.95,
                          double theta_min = -0.95,
                          double theta_max = 0.95,
                          double step = 0.02);

std::vector<double> conditional_innovations(const std::vector<double>& x,
                                            const ARMA11Parameters& params);

} // namespace ts
