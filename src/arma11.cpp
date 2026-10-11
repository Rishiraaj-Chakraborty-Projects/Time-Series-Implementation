#include "ts/arma11.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace ts {

std::vector<double> simulate_arma11(const ARMA11Parameters& params,
                                     std::size_t n,
                                     unsigned int seed,
                                     std::size_t burn_in) {
    if (n == 0) {
        return {};
    }
    if (params.sigma <= 0.0) {
        throw std::invalid_argument("sigma must be positive");
    }
    if (std::abs(params.phi) >= 1.0) {
        throw std::invalid_argument("Simulation requires |phi| < 1 for stationarity");
    }

    std::mt19937 generator(seed);
    std::normal_distribution<double> shock(0.0, params.sigma);

    std::vector<double> values(n + burn_in, params.intercept / (1.0 - params.phi));
    double previous_shock = 0.0;

    for (std::size_t t = 1; t < values.size(); ++t) {
        const double current_shock = shock(generator);
        values[t] = params.intercept
                  + params.phi * values[t - 1]
                  + current_shock
                  + params.theta * previous_shock;
        previous_shock = current_shock;
    }

    values.erase(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(burn_in));
    return values;
}

double theoretical_mean(const ARMA11Parameters& params) {
    if (std::abs(params.phi) >= 1.0) {
        throw std::invalid_argument("Theoretical mean requires |phi| < 1");
    }
    return params.intercept / (1.0 - params.phi);
}

double theoretical_variance(const ARMA11Parameters& params) {
    if (std::abs(params.phi) >= 1.0 || params.sigma <= 0.0) {
        throw std::invalid_argument("Invalid stationary ARMA(1,1) parameters");
    }

    const double phi = params.phi;
    const double theta = params.theta;
    const double sigma_sq = params.sigma * params.sigma;

    // gamma_0 = sigma^2 (1 + theta^2 + 2 phi theta) / (1 - phi^2)
    return sigma_sq * (1.0 + theta * theta + 2.0 * phi * theta)
         / (1.0 - phi * phi);
}

std::vector<double> conditional_innovations(const std::vector<double>& x,
                                            const ARMA11Parameters& params) {
    if (x.size() < 2) {
        throw std::invalid_argument("Need at least two observations");
    }
    if (std::abs(params.phi) >= 1.0) {
        throw std::invalid_argument("Conditional innovations require |phi| < 1");
    }

    std::vector<double> errors(x.size(), 0.0);
    // The first residual is initialized from the unconditional mean.
    errors[0] = x[0] - theoretical_mean(params);

    for (std::size_t t = 1; t < x.size(); ++t) {
        errors[t] = x[t]
                    - params.intercept
                    - params.phi * x[t - 1]
                    - params.theta * errors[t - 1];
    }

    return errors;
}

ARMA11Fit fit_arma11_grid(const std::vector<double>& x,
                          double phi_min,
                          double phi_max,
                          double theta_min,
                          double theta_max,
                          double step) {
    if (x.size() < 20) {
        throw std::invalid_argument("Need at least 20 observations for the grid fit");
    }
    if (!(phi_min < phi_max) || !(theta_min < theta_max) || step <= 0.0) {
        throw std::invalid_argument("Invalid grid-search bounds");
    }

    const double mu = std::accumulate(x.begin(), x.end(), 0.0) / static_cast<double>(x.size());
    const double intercept = mu * 0.5; // overwritten below using each candidate phi

    (void)intercept;

    ARMA11Fit best;
    best.objective = std::numeric_limits<double>::infinity();
    best.observations = x.size();

    for (double phi = phi_min; phi <= phi_max + step * 0.25; phi += step) {
        // Profile out the intercept through the sample mean relation c = mu(1 - phi).
        const double candidate_intercept = mu * (1.0 - phi);

        for (double theta = theta_min; theta <= theta_max + step * 0.25; theta += step) {
            ARMA11Parameters candidate;
            candidate.phi = phi;
            candidate.theta = theta;
            candidate.intercept = candidate_intercept;
            candidate.sigma = 1.0;

            const auto errors = conditional_innovations(x, candidate);
            double sse = 0.0;
            // Ignore the first conditional residual because its initialization is approximate.
            for (std::size_t t = 1; t < errors.size(); ++t) {
                sse += errors[t] * errors[t];
            }

            if (sse < best.objective) {
                best.objective = sse;
                best.parameters = candidate;
            }
        }
    }

    best.parameters.sigma = std::sqrt(best.objective
                                      / static_cast<double>(x.size() - 1));
    return best;
}

} // namespace ts
