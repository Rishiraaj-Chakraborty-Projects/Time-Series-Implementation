#include "ts/arma11.hpp"
#include "ts/csv.hpp"
#include "ts/diagnostics.hpp"
#include "ts/forecast.hpp"
#include "ts/statistics.hpp"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <vector>

#define TEST_CHECK(expr)                                                       \
    do {                                                                       \
        if (!(expr)) {                                                         \
            std::cerr << "FAIL: " << #expr << " at " << __FILE__ << ":"        \
                      << __LINE__ << std::endl;                                \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

#define TEST_NEAR(actual, expected, tol)                                       \
    do {                                                                       \
        const double a_ = (actual);                                            \
        const double e_ = (expected);                                          \
        const double diff_ = std::abs(a_ - e_);                                \
        if (diff_ > (tol)) {                                                   \
            std::cerr << "FAIL: " << #actual << " (" << a_                     \
                      << ") != " << #expected << " (" << e_                    \
                      << ") [diff=" << diff_ << " > tol=" << (tol) << "] at " \
                      << __FILE__ << ":" << __LINE__ << std::endl;             \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

#define TEST_THROWS(expr, ExceptionType)                                       \
    do {                                                                       \
        bool caught_ = false;                                                  \
        try {                                                                  \
            expr;                                                              \
        } catch (const ExceptionType&) {                                       \
            caught_ = true;                                                    \
        } catch (...) {                                                        \
        }                                                                      \
        if (!caught_) {                                                        \
            std::cerr << "FAIL: Expected " << #ExceptionType << " for "        \
                      << #expr << " at " << __FILE__ << ":"                    \
                      << __LINE__ << std::endl;                                \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

void test_statistics() {
    std::vector<double> x{1.0, 2.0, 3.0, 4.0};
    TEST_NEAR(ts::mean(x), 2.5, 1e-12);
    TEST_NEAR(ts::variance(x, true), 5.0 / 3.0, 1e-12);
    TEST_NEAR(ts::variance(x, false), 1.25, 1e-12);
    TEST_NEAR(ts::standard_deviation(x, true), std::sqrt(5.0 / 3.0), 1e-12);
    TEST_NEAR(ts::autocovariance(x, 0), 1.25, 1e-12);
    TEST_NEAR(ts::autocovariance(x, 1), 0.3125, 1e-12);
    TEST_NEAR(ts::autocorrelation(x, 0), 1.0, 1e-12);
    TEST_NEAR(ts::autocorrelation(x, 1), 0.25, 1e-12);

    const auto acf_vals = ts::acf(x, 2);
    TEST_CHECK(acf_vals.size() == 3);
    TEST_NEAR(acf_vals[0], 1.0, 1e-12);
    TEST_NEAR(acf_vals[1], 0.25, 1e-12);

    const auto summary = ts::summarize(x);
    TEST_CHECK(summary.observations == 4);
    TEST_NEAR(summary.mean, 2.5, 1e-12);
    TEST_NEAR(summary.variance, 5.0 / 3.0, 1e-12);

    // Constant series zero-variance handling
    std::vector<double> constant_x{4.0, 4.0, 4.0, 4.0};
    TEST_NEAR(ts::variance(constant_x, true), 0.0, 1e-12);
    TEST_NEAR(ts::autocorrelation(constant_x, 0), 1.0, 1e-12);
    TEST_NEAR(ts::autocorrelation(constant_x, 1), 0.0, 1e-12);

    // Exception handling
    std::vector<double> empty_x{};
    TEST_THROWS(ts::mean(empty_x), std::invalid_argument);
    TEST_THROWS(ts::variance(empty_x), std::invalid_argument);
    TEST_THROWS(ts::autocovariance(x, 5), std::invalid_argument);
}

void test_arma11_core() {
    ts::ARMA11Parameters p;
    p.phi = 0.8;
    p.theta = 0.2;
    p.intercept = 0.4;
    p.sigma = 2.0;

    const double expected_mean = 2.0;
    TEST_NEAR(ts::theoretical_mean(p), expected_mean, 1e-12);

    const double expected_variance =
        4.0 * (1.0 + 0.04 + 2.0 * 0.8 * 0.2) / (1.0 - 0.64);
    TEST_NEAR(ts::theoretical_variance(p), expected_variance, 1e-12);

    const auto simulated = ts::simulate_arma11(p, 500, 1234);
    TEST_CHECK(simulated.size() == 500);

    // Non-stationary / invalid parameter checks
    ts::ARMA11Parameters non_stationary = p;
    non_stationary.phi = 1.0;
    TEST_THROWS(ts::theoretical_mean(non_stationary), std::invalid_argument);
    TEST_THROWS(ts::theoretical_variance(non_stationary), std::invalid_argument);
    TEST_THROWS(ts::simulate_arma11(non_stationary, 100), std::invalid_argument);

    ts::ARMA11Parameters bad_sigma = p;
    bad_sigma.sigma = -1.0;
    TEST_THROWS(ts::theoretical_variance(bad_sigma), std::invalid_argument);
    TEST_THROWS(ts::simulate_arma11(bad_sigma, 100), std::invalid_argument);
}

void test_conditional_innovations() {
    ts::ARMA11Parameters p;
    p.phi = 0.5;
    p.theta = 0.2;
    p.intercept = 1.0;
    p.sigma = 1.0;

    // theoretical_mean = 1.0 / (1.0 - 0.5) = 2.0
    std::vector<double> x{2.0, 2.5, 3.0};
    const auto err = ts::conditional_innovations(x, p);
    TEST_CHECK(err.size() == 3);

    // err[0] = x[0] - mu = 2.0 - 2.0 = 0.0
    TEST_NEAR(err[0], 0.0, 1e-12);
    // err[1] = 2.5 - 1.0 - 0.5*2.0 - 0.2*0.0 = 0.5
    TEST_NEAR(err[1], 0.5, 1e-12);
    // err[2] = 3.0 - 1.0 - 0.5*2.5 - 0.2*0.5 = 2.0 - 1.25 - 0.10 = 0.65
    TEST_NEAR(err[2], 0.65, 1e-12);

    std::vector<double> single_x{1.0};
    TEST_THROWS(ts::conditional_innovations(single_x, p), std::invalid_argument);
}

void test_forecasting() {
    ts::ARMA11Parameters p;
    p.phi = 0.5;
    p.theta = 0.2;
    p.intercept = 1.0;
    p.sigma = 1.0;

    std::vector<double> x{2.0, 2.5, 3.0};
    const auto fc0 = ts::forecast_arma11(x, p, 0);
    TEST_CHECK(fc0.empty());

    const auto fc3 = ts::forecast_arma11(x, p, 3);
    TEST_CHECK(fc3.size() == 3);

    // h=1: c + phi * x[2] + theta * err[2] = 1.0 + 0.5*3.0 + 0.2*0.65 = 1.0 + 1.5 + 0.13 = 2.63
    TEST_NEAR(fc3[0], 2.63, 1e-12);
    // h=2: c + phi * fc3[0] = 1.0 + 0.5 * 2.63 = 2.315
    TEST_NEAR(fc3[1], 2.315, 1e-12);
    // h=3: c + phi * fc3[1] = 1.0 + 0.5 * 2.315 = 2.1575
    TEST_NEAR(fc3[2], 2.1575, 1e-12);

    // Long horizon should approach theoretical mean = 2.0
    const auto fc_long = ts::forecast_arma11(x, p, 50);
    TEST_NEAR(fc_long.back(), 2.0, 1e-4);

    TEST_THROWS(ts::forecast_arma11({}, p, 5), std::invalid_argument);
}

void test_grid_fit() {
    ts::ARMA11Parameters true_p;
    true_p.phi = 0.6;
    true_p.theta = -0.3;
    true_p.intercept = 1.2;
    true_p.sigma = 0.8;

    const auto series = ts::simulate_arma11(true_p, 400, 9999);
    const auto fit = ts::fit_arma11_grid(series, 0.4, 0.8, -0.6, 0.0, 0.05);

    TEST_CHECK(fit.observations == 400);
    TEST_CHECK(fit.objective > 0.0);
    // Grid fit should recover parameters reasonably close to the truth
    TEST_CHECK(std::abs(fit.parameters.phi - true_p.phi) < 0.2);
    TEST_CHECK(std::abs(fit.parameters.theta - true_p.theta) < 0.25);

    std::vector<double> small_series(10, 1.0);
    TEST_THROWS(ts::fit_arma11_grid(small_series), std::invalid_argument);
}

void test_diagnostics() {
    std::vector<double> residuals{0.1, -0.2, 0.05, 0.12, -0.08, 0.03, -0.15, 0.07, -0.02, 0.04};
    const auto lb = ts::ljung_box(residuals, 3);
    TEST_CHECK(lb.lags == 3);
    TEST_CHECK(lb.statistic >= 0.0);
    TEST_CHECK(lb.approximate_p_value >= 0.0 && lb.approximate_p_value <= 1.0);

    TEST_THROWS(ts::ljung_box(residuals, 0), std::invalid_argument);
    std::vector<double> few_residuals{0.1, 0.2};
    TEST_THROWS(ts::ljung_box(few_residuals, 1), std::invalid_argument);
}

void test_csv_io() {
    const std::string temp_file = "test_temp_series.csv";
    std::vector<double> original{10.5, 20.25, 30.125, 40.0625};

    ts::write_series_csv(temp_file, original, "test_col");
    const auto loaded = ts::read_single_column_csv(temp_file, true);

    TEST_CHECK(loaded.values.size() == original.size());
    for (std::size_t i = 0; i < original.size(); ++i) {
        TEST_NEAR(loaded.values[i], original[i], 1e-12);
    }

    std::filesystem::remove(temp_file);

    TEST_THROWS(ts::read_single_column_csv("non_existent_file_xyz.csv"), std::runtime_error);
}

int main() {
    std::cout << "Running Time Series Lab test suite...\n";

    test_statistics();
    std::cout << "  [PASS] test_statistics\n";

    test_arma11_core();
    std::cout << "  [PASS] test_arma11_core\n";

    test_conditional_innovations();
    std::cout << "  [PASS] test_conditional_innovations\n";

    test_forecasting();
    std::cout << "  [PASS] test_forecasting\n";

    test_grid_fit();
    std::cout << "  [PASS] test_grid_fit\n";

    test_diagnostics();
    std::cout << "  [PASS] test_diagnostics\n";

    test_csv_io();
    std::cout << "  [PASS] test_csv_io\n";

    std::cout << "\nAll tests passed successfully.\n";
    return 0;
}
