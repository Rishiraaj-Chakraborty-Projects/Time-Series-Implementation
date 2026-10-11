#pragma once

#include <cstddef>
#include <vector>

namespace ts {

struct LjungBoxResult {
    double statistic{};
    double approximate_p_value{};
    std::size_t lags{};
};

LjungBoxResult ljung_box(const std::vector<double>& residuals, std::size_t lags);

} // namespace ts
