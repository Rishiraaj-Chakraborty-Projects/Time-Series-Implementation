#pragma once

#include <cstddef>
#include <vector>

#include "ts/arma11.hpp"

namespace ts {

std::vector<double> forecast_arma11(const std::vector<double>& x,
                                    const ARMA11Parameters& params,
                                    std::size_t horizon);

} // namespace ts
