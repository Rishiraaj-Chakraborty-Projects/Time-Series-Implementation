#include "ts/statistics.hpp"

#include <cmath>
#include <numeric>
#include <stdexcept>

namespace ts {

double mean(const std::vector<double>& x) {
    if (x.empty()) {
        throw std::invalid_argument("mean() requires at least one observation");
    }
    return std::accumulate(x.begin(), x.end(), 0.0) / static_cast<double>(x.size());
}

double variance(const std::vector<double>& x, bool sample) {
    if (x.size() < (sample ? 2u : 1u)) {
        throw std::invalid_argument("variance() received too few observations");
    }

    const double mu = mean(x);
    double sum_sq = 0.0;
    for (double value : x) {
        const double deviation = value - mu;
        sum_sq += deviation * deviation;
    }

    const double denominator = sample ? static_cast<double>(x.size() - 1)
                                      : static_cast<double>(x.size());
    return sum_sq / denominator;
}

double standard_deviation(const std::vector<double>& x, bool sample) {
    return std::sqrt(variance(x, sample));
}

double autocovariance(const std::vector<double>& x, std::size_t lag) {
    if (x.empty() || lag >= x.size()) {
        throw std::invalid_argument("autocovariance() received an invalid lag");
    }

    const double mu = mean(x);
    double total = 0.0;
    for (std::size_t t = lag; t < x.size(); ++t) {
        total += (x[t] - mu) * (x[t - lag] - mu);
    }

    return total / static_cast<double>(x.size());
}

double autocorrelation(const std::vector<double>& x, std::size_t lag) {
    const double gamma0 = autocovariance(x, 0);
    if (gamma0 == 0.0) {
        return lag == 0 ? 1.0 : 0.0;
    }
    return autocovariance(x, lag) / gamma0;
}

std::vector<double> acf(const std::vector<double>& x, std::size_t max_lag) {
    if (max_lag >= x.size()) {
        max_lag = x.size() - 1;
    }

    std::vector<double> result;
    result.reserve(max_lag + 1);
    for (std::size_t lag = 0; lag <= max_lag; ++lag) {
        result.push_back(autocorrelation(x, lag));
    }
    return result;
}

SummaryStats summarize(const std::vector<double>& x) {
    SummaryStats stats;
    stats.observations = x.size();
    stats.mean = mean(x);
    stats.variance = variance(x);
    stats.standard_deviation = std::sqrt(stats.variance);
    return stats;
}

} // namespace ts
