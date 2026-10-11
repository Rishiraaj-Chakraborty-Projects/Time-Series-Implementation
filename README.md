# Time Series Lab — ARMA(1,1) in Modern C++

A small, dependency-light C++17 time-series project built around one idea: **implement the mathematics yourself and make every step inspectable.**

The project simulates, analyzes, estimates, diagnoses, and forecasts an ARMA(1,1) process without pulling in a large numerical/statistics framework.

## Why this project exists

The goal is not to compete with R, Python, statsmodels, or a professional numerical library. The goal is to demonstrate that I understand what the time-series machinery is doing underneath the library call.

The central model is:

```text
X_t = c + phi * X_(t-1) + eta_t + theta * eta_(t-1)
```

where `eta_t` is white noise.

## Features

- ARMA(1,1) simulation with Gaussian shocks
- Stationary theoretical mean
- Stationary theoretical variance
- Mean, variance, standard deviation
- Autocovariance and ACF calculation
- Conditional innovation calculation
- Transparent ARMA(1,1) grid-search estimation
- Multi-step forecasting
- Lightweight residual diagnostic
- CSV input/output
- Unit tests
- CMake build
- No third-party runtime dependencies

## Project structure

```text
TimeSeriesLab/
├── CMakeLists.txt
├── README.md
├── docs/
│   └── mathematics.md
├── data/
│   └── example_series.csv
├── include/ts/
│   ├── arma11.hpp
│   ├── csv.hpp
│   ├── diagnostics.hpp
│   ├── forecast.hpp
│   └── statistics.hpp
├── src/
│   ├── arma11.cpp
│   ├── csv.cpp
│   ├── diagnostics.cpp
│   ├── forecast.cpp
│   ├── main.cpp
│   └── statistics.cpp
└── tests/
    └── test_main.cpp
```

The split is deliberate: `main.cpp` handles the CLI, while the actual model/statistics logic lives in reusable library code.

## Build

### Linux / macOS

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

### Windows with Visual Studio / Developer PowerShell

```powershell
cmake -S . -B build
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

You need a C++17 compiler and CMake 3.16+.

## Run it

Generate a synthetic ARMA(1,1) series:

```bash
./build/ts_lab simulate data/simulated_arma11.csv
```

Windows Visual Studio builds commonly put the executable under:

```text
build/Release/ts_lab.exe
```

Analyze a CSV containing a single numeric column:

```bash
./build/ts_lab analyze data/simulated_arma11.csv
```

Forecast the next 10 observations:

```bash
./build/ts_lab forecast data/simulated_arma11.csv 10
```

## What the estimator is doing

For each candidate `phi` and `theta`, the program reconstructs the innovations recursively:

```text
eta_hat[t] = X[t]
            - c
            - phi * X[t-1]
            - theta * eta_hat[t-1]
```

It then calculates the sum of squared innovations. The candidate with the smallest SSE is selected.

This is intentionally simple enough that a recruiter can read the code and understand the implementation in one sitting.

## Important limitation

The ARMA fitting routine is a **transparent conditional sum-of-squares grid search**, not a full industrial maximum-likelihood estimator. The README says this explicitly because the project is intended to show understanding rather than pretend to be a production statistics package.

The residual p-value shown by the lightweight diagnostic is also explicitly approximate. For serious statistical inference, a dedicated numerical/statistics library should be used.

## Recruiter talking points

A good way to discuss this project is:

> "I implemented an ARMA(1,1) pipeline in C++17 from the equations rather than wrapping a statistics library. I separated simulation, statistics, estimation, forecasting, diagnostics, and CSV I/O into different translation units. The estimator reconstructs conditional innovations recursively and searches over phi and theta by conditional sum of squares."

Then explain the most important mathematical point:

> "The MA term affects the current and recent shock response, while the AR term creates persistence. In the variance derivation, the cross-term appears because the previous state contains the previous innovation."

## Next extensions

Natural next steps if this becomes a larger research project:

1. Implement exact Gaussian maximum likelihood.
2. Add AR(p) and MA(q) classes.
3. Add AIC/BIC model selection.
4. Add confidence intervals through bootstrap or asymptotic approximations.
5. Add real financial data and rolling out-of-sample forecasts.
6. Add volatility models such as GARCH(1,1).
7. Add a small benchmarking suite against a trusted statistics library.

That progression turns this from a learning project into a small time-series research framework.
