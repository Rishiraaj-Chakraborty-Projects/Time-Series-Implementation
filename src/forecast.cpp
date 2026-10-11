#include "ts/forecast.hpp"

#include <stdexcept>

namespace ts {

std::vector<double> forecast_arma11(const std::vector<double>& x,
                                    const ARMA11Parameters& params,
                                    std::size_t horizon) {
    if (x.size() < 2) {
        throw std::invalid_argument("forecast_arma11() requires at least two observations");
    }
    if (horizon == 0) {
        return {};
    }

    const auto errors = conditional_innovations(x, params);
    double last_value = x.back();
    double last_error = errors.back();

    std::vector<double> forecast;
    forecast.reserve(horizon);

    // E[eta_{T+h} | information through T] = 0 for h >= 1.
    // Only the known last innovation contributes at step h = 1.
    const double first = params.intercept
                       + params.phi * last_value
                       + params.theta * last_error;
    forecast.push_back(first);

    for (std::size_t h = 1; h < horizon; ++h) {
        const double next = params.intercept + params.phi * forecast.back();
        forecast.push_back(next);
    }

    return forecast;
}

} // namespace ts
