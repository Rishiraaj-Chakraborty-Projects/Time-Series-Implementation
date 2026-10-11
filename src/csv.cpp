#include "ts/csv.hpp"

#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace ts {

TimeSeries read_single_column_csv(const std::string& path, bool has_header) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("Could not open CSV file: " + path);
    }

    TimeSeries series;
    std::string line;
    bool first_line = true;

    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty()) {
            continue;
        }
        if (first_line && has_header) {
            first_line = false;
            continue;
        }
        first_line = false;

        std::stringstream ss(line);
        std::string field;
        if (!std::getline(ss, field, ',')) {
            continue;
        }
        try {
            series.values.push_back(std::stod(field));
        } catch (const std::exception&) {
            throw std::runtime_error("Invalid numeric value in CSV: " + field);
        }
    }

    if (series.values.empty()) {
        throw std::runtime_error("CSV contained no observations: " + path);
    }
    return series;
}

void write_series_csv(const std::string& path,
                      const std::vector<double>& values,
                      const std::string& header) {
    std::ofstream file(path);
    if (!file) {
        throw std::runtime_error("Could not create CSV file: " + path);
    }

    file << header << '\n';
    file << std::setprecision(15);
    for (double value : values) {
        file << value << '\n';
    }
}

} // namespace ts
