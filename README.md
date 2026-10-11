# Time Series Lab — ARMA(1,1) in Modern C++

![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)
![CMake](https://img.shields.io/badge/CMake-3.16%2B-064F8C.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Build](https://img.shields.io/badge/Tests-Passing-brightgreen.svg)

A clean, dependency-free C++17 library and CLI tool built from first principles to simulate, analyze, fit, diagnose, and forecast autoregressive moving-average $\text{ARMA}(1,1)$ processes.

The goal of this project is to implement core time-series algorithms and statistical derivations directly from the mathematics rather than delegating to heavy external frameworks, keeping every calculation transparent, deterministic, and inspectable.

---

## The Model

The canonical $\text{ARMA}(1,1)$ specification is:

$$X_t = c + \phi X_{t-1} + \eta_t + \theta \eta_{t-1}, \quad \eta_t \overset{\text{i.i.d.}}{\sim} \mathcal{N}(0, \sigma^2)$$

Where:
* $\phi$ is the autoregressive coefficient, governing process persistence (stationarity requires $|\phi| < 1$).
* $\theta$ is the moving-average coefficient, weighting recent shock memory (invertibility requires $|\theta| < 1$).
* $c$ is the drift/intercept parameter, tied to the unconditional mean by $\mu = \frac{c}{1 - \phi}$.
* $\eta_t$ represents the Gaussian white-noise innovation at time $t$.

---

## Features

* **Simulation Engine**: Generates synthetic $\text{ARMA}(1,1)$ paths with configurable burn-in to eliminate initial-state transient effects.
* **Exact Analytical Properties**: Evaluates theoretical stationary mean and variance derived directly from model expectations.
* **Sample Statistics**: Efficient single-pass sample mean, sample/unbiased variance, sample autocovariance, and sample autocorrelation function (ACF).
* **Conditional Sum-of-Squares (CSS) Fitting**: Bounded grid search across the stationary and invertible parameter space. Profiles out the intercept via sample mean to reduce search dimensionality.
* **Zero-Allocation Error Evaluation**: Reconstructs conditional innovations in-place during parameter optimization to ensure cache-friendly performance.
* **Multi-Step Forecasting**: Computes recursive conditional expectations $\mathbb{E}[X_{T+h} \mid \mathcal{F}_T]$, which decay smoothly toward the unconditional stationary mean as the horizon expands.
* **Residual Diagnostics**: Computes Ljung-Box portmanteau test statistics with Wilson-Hilferty transformation for Chi-square survival probabilities without external numerical libraries.
* **Lightweight CSV I/O**: Streamlined parser and writer for univariate time-series data.
* **Zero External Dependencies**: Implemented entirely with the C++17 standard library (`<vector>`, `<cmath>`, `<random>`, `<filesystem>`).

---

## Project Structure

```text
time_series_project/
├── CMakeLists.txt              # Multi-target CMake build with strict compiler warnings (-Wall, /W4)
├── README.md                   # Project overview, architecture, and usage
├── LICENSE                     # MIT License
├── docs/
│   └── mathematics.md          # Formal mathematical derivations (mean, variance, CSS, forecasting)
├── data/
│   ├── example_series.csv      # Sample empirical series for demonstration
│   └── simulated.csv           # Reference synthetic dataset
├── include/ts/                 # Public header interfaces
│   ├── arma11.hpp              # ARMA(1,1) model definitions, simulation, and CSS fitter
│   ├── csv.hpp                 # Fast univariate CSV I/O
│   ├── diagnostics.hpp         # Ljung-Box test and goodness-of-fit diagnostics
│   ├── forecast.hpp            # Multi-step forecasting interface
│   └── statistics.hpp          # Mean, variance, autocovariance, and ACF functions
├── src/                        # Library implementation and CLI
│   ├── arma11.cpp
│   ├── csv.cpp
│   ├── diagnostics.cpp
│   ├── forecast.cpp
│   ├── main.cpp                # Command-line interface driver
│   └── statistics.cpp
└── tests/
    └── test_main.cpp           # Deterministic unit test suite covering edge cases and error bounds
```

The core computational logic lives in `ts_core` as a standalone static library, cleanly decoupled from the `ts_lab` CLI entry point.

---

## Building and Testing

### Prerequisites
* C++17 compliant compiler (GCC 9+, Clang 10+, or MSVC 2019+)
* CMake 3.16+

### Linux / macOS

```bash
# Configure and build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# Execute test suite
ctest --test-dir build --output-on-failure
```

### Windows (Visual Studio / PowerShell)

```powershell
# Configure and build Release binaries
cmake -S . -B build
cmake --build build --config Release

# Execute test suite
ctest --test-dir build -C Release --output-on-failure
```

---

## Usage Guide

The compiled executable `ts_lab` provides three primary commands:

### 1. Simulate Synthetic Series
Generates 500 observations from a known parameter set ($\phi = 0.75, \theta = 0.35, c = 0.50, \sigma = 1.0$) and exports to CSV:

```bash
# Linux / macOS
./build/ts_lab simulate data/simulated_arma11.csv

# Windows
.\build\Release\ts_lab.exe simulate data\simulated_arma11.csv
```

### 2. Analyze and Estimate Parameters
Parses a univariate series, computes summary statistics and sample ACF, fits the $\text{ARMA}(1,1)$ model via conditional sum-of-squares, and runs Ljung-Box residual diagnostics:

```bash
./build/ts_lab analyze data/example_series.csv
```

**Example Output:**
```text
Observations: 60
Mean:        8.546000
Variance:    10.490750
Std. dev.:   3.238943

ACF
lag	correlation
0	1.00000
1	0.93173
2	0.88548
...
10	0.49618

Fitting ARMA(1,1) by conditional sum of squared innovations...
phi:         0.95000
theta:       -0.16000
intercept:   0.42730
sigma:       0.54296
SSE:         17.39332

Ljung-Box diagnostic test (lag 10):
Q-statistic: 4.61060
p-value:     0.91572
```

### 3. Forecast Future Horizons
Fits model parameters on the input series and projects the conditional mean $h$ steps ahead:

```bash
./build/ts_lab forecast data/example_series.csv 10
```

---

## Methodology & Engineering Decisions

### Conditional Innovations & In-Place SSE
Given observations $X_1, \dots, X_T$ and trial coefficients $(\phi, \theta, c)$, innovations are computed recursively:

$$\hat{\eta}_t = X_t - c - \phi X_{t-1} - \theta \hat{\eta}_{t-1}, \quad t \ge 2$$

Initializing $\hat{\eta}_1 = X_1 - \mu$. The grid search evaluates the parameter surface by minimizing:

$$\text{SSE}(\phi, \theta) = \sum_{t=2}^T \hat{\eta}_t^2$$

To ensure high performance over tens of thousands of candidate grid points, the inner loop computes the error recurrence using scalar registers without intermediate vector allocations.

### Intercept Profiling
Rather than searching across a three-dimensional $(\phi, \theta, c)$ cube, the intercept is profiled out analytically using the sample mean identity:

$$c = \bar{X}(1 - \phi)$$

This reduces search complexity to two dimensions while guaranteeing that candidate models preserve the sample mean.

### Mathematical Derivations
For detailed mathematical proofs — including the stationary variance cross-term derivation ($\gamma_0 = \frac{\sigma^2(1 + \theta^2 + 2\phi\theta)}{1 - \phi^2}$) and multi-step forecast variance — see [docs/mathematics.md](docs/mathematics.md).

---

## Future Extensions

1. **Exact Gaussian MLE**: Kalman-filter state-space formulation to evaluate the exact likelihood function including the initial diffuse state.
2. **General $\text{ARMA}(p, q)$**: Extend state equations to arbitrary lag orders using companion matrix representation.
3. **Information Criteria**: Implement AIC and BIC penalties for automated lag-order selection.
4. **Confidence Bounds**: Asymptotic standard errors via numerical Hessian computation at the optimum.
5. **ARCH / GARCH**: Conditional heteroskedasticity modeling for volatility clustering.

---

## License

This project is open-source and available under the [MIT License](LICENSE).
