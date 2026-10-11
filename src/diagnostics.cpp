#include "ts/diagnostics.hpp"

#include "ts/statistics.hpp"

#include <cmath>
#include <stdexcept>

namespace ts {

LjungBoxResult ljung_box(const std::vector<double>& residuals, std::size_t lags) {
    if (residuals.size() < 10) {
        throw std::invalid_argument("Need at least 10 residuals");
    }
    if (lags == 0) {
        throw std::invalid_argument("Ljung-Box needs at least one lag");
    }

    lags = std::min(lags, residuals.size() - 1);

    double q = 0.0;
    const double n = static_cast<double>(residuals.size());

    for (std::size_t k = 1; k <= lags; ++k) {
        const double rho = autocorrelation(residuals, k);
        q += (rho * rho) / (n - static_cast<double>(k));
    }
    q *= n * (n + 2.0);

    // For a lightweight, dependency-free project we use a rough chi-square
    // tail approximation. It is intended as a diagnostic, not a stats library.
    const double df = static_cast<double>(lags);
    const double p = std::exp(-0.5 * q / std::max(df, 1.0));

    return {q, p, lags};
}

} // namespace ts
