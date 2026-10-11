#pragma once

#include <string>
#include <vector>

namespace ts {

struct TimeSeries {
    std::vector<double> values;
};

TimeSeries read_single_column_csv(const std::string& path, bool has_header = true);
void write_series_csv(const std::string& path,
                      const std::vector<double>& values,
                      const std::string& header = "value");

} // namespace ts
