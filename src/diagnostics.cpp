#include "ts/diagnostics.hpp"

#include "ts/statistics.hpp"

#include <algorithm>
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

    // Wilson-Hilferty transformation: approximates Chi-square(df) survival function
    // via transformation to the standard normal distribution without requiring external gamma libraries.
    const double df = static_cast<double>(lags);
    double p = 0.0;
    if (q <= 0.0) {
        p = 1.0;
    } else {
        const double factor = 2.0 / (9.0 * df);
        const double z = (std::cbrt(q / df) - (1.0 - factor)) / std::sqrt(factor);
        // Survival function: 1 - Phi(z) = 0.5 * erfc(z / sqrt(2))
        p = 0.5 * std::erfc(z / std::sqrt(2.0));
        p = std::clamp(p, 0.0, 1.0);
    }

    return {q, p, lags};
}

} // namespace ts
