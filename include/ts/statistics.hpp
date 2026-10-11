#pragma once

#include <vector>

namespace ts {

double mean(const std::vector<double>& x);
double variance(const std::vector<double>& x, bool sample = true);
double standard_deviation(const std::vector<double>& x, bool sample = true);

double autocovariance(const std::vector<double>& x, std::size_t lag);
double autocorrelation(const std::vector<double>& x, std::size_t lag);
std::vector<double> acf(const std::vector<double>& x, std::size_t max_lag);

struct SummaryStats {
    std::size_t observations{};
    double mean{};
    double variance{};
    double standard_deviation{};
};

SummaryStats summarize(const std::vector<double>& x);

} // namespace ts
