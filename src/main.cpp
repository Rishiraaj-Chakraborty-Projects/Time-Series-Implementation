#include "ts/arma11.hpp"
#include "ts/csv.hpp"
#include "ts/diagnostics.hpp"
#include "ts/forecast.hpp"
#include "ts/statistics.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

namespace {

void print_usage(const char* program) {
    std::cout
        << "Time Series Lab - ARMA(1,1) demonstration\n\n"
        << "Usage:\n"
        << "  " << program << " simulate [output.csv]\n"
        << "  " << program << " analyze <input.csv>\n"
        << "  " << program << " forecast <input.csv> [horizon]\n";
}

void print_acf(const std::vector<double>& values, std::size_t max_lag) {
    const auto values_acf = ts::acf(values, max_lag);
    std::cout << "\nACF\n"
              << "lag\tcorrelation\n";
    std::cout << std::fixed << std::setprecision(5);
    for (std::size_t lag = 0; lag < values_acf.size(); ++lag) {
        std::cout << lag << '\t' << values_acf[lag] << '\n';
    }
}

int simulate(const std::string& output_path) {
    ts::ARMA11Parameters params;
    params.phi = 0.75;
    params.theta = 0.35;
    params.intercept = 0.50;
    params.sigma = 1.0;

    const auto series = ts::simulate_arma11(params, 500, 20261003);
    ts::write_series_csv(output_path, series);

    std::cout << "Generated 500 observations from:\n"
              << "  X_t = c + phi X_(t-1) + eta_t + theta eta_(t-1)\n"
              << "  c     = " << params.intercept << '\n'
              << "  phi   = " << params.phi << '\n'
              << "  theta = " << params.theta << '\n'
              << "  sigma = " << params.sigma << '\n'
              << "\nTheoretical mean: " << ts::theoretical_mean(params)
              << "\nTheoretical variance: " << ts::theoretical_variance(params)
              << "\n\nSaved to: " << output_path << '\n';
    return 0;
}

int analyze(const std::string& input_path) {
    const auto series = ts::read_single_column_csv(input_path);
    const auto stats = ts::summarize(series.values);

    std::cout << std::fixed << std::setprecision(6)
              << "Observations: " << stats.observations << '\n'
              << "Mean:        " << stats.mean << '\n'
              << "Variance:    " << stats.variance << '\n'
              << "Std. dev.:   " << stats.standard_deviation << '\n';

    print_acf(series.values, 12);

    std::cout << "\nFitting ARMA(1,1) by conditional sum of squared innovations...\n";
    const auto fit = ts::fit_arma11_grid(series.values);
    std::cout << "phi:         " << fit.parameters.phi << '\n'
              << "theta:       " << fit.parameters.theta << '\n'
              << "intercept:   " << fit.parameters.intercept << '\n'
              << "sigma:       " << fit.parameters.sigma << '\n'
              << "SSE:         " << fit.objective << '\n';

    const auto residuals = ts::conditional_innovations(series.values, fit.parameters);
    const auto lb = ts::ljung_box(residuals, 10);
    std::cout << "\nLjung-Box diagnostic test (lag 10):\n"
              << "Q-statistic: " << lb.statistic << '\n'
              << "p-value:     " << lb.approximate_p_value << '\n';

    return 0;
}

int forecast(const std::string& input_path, std::size_t horizon) {
    const auto series = ts::read_single_column_csv(input_path);
    const auto fit = ts::fit_arma11_grid(series.values);
    const auto predictions = ts::forecast_arma11(series.values, fit.parameters, horizon);

    std::cout << std::fixed << std::setprecision(6)
              << "Fitted model: phi=" << fit.parameters.phi
              << ", theta=" << fit.parameters.theta
              << ", intercept=" << fit.parameters.intercept << '\n'
              << "\nForecast\nhorizon\tprediction\n";

    for (std::size_t h = 0; h < predictions.size(); ++h) {
        std::cout << (h + 1) << '\t' << predictions[h] << '\n';
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            print_usage(argv[0]);
            return 1;
        }

        const std::string command = argv[1];
        if (command == "simulate") {
            return simulate(argc >= 3 ? argv[2] : "data/simulated_arma11.csv");
        }
        if (command == "analyze" && argc >= 3) {
            return analyze(argv[2]);
        }
        if (command == "forecast" && argc >= 3) {
            const std::size_t horizon = argc >= 4 ? std::stoul(argv[3]) : 10;
            return forecast(argv[2], horizon);
        }

        print_usage(argv[0]);
        return 1;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 2;
    }
}
