# Mathematics behind the project

## 1. ARMA(1,1)

The model is

\[
X_t = c + \phi X_{t-1} + \eta_t + \theta\eta_{t-1},
\]

where the shocks satisfy

\[
E[\eta_t]=0, \qquad Var(\eta_t)=\sigma^2,
\]

and different shocks are uncorrelated.

`phi` controls persistence. `theta` changes the way the current observation responds to the current and previous shocks.

## 2. Mean

For a stationary process, \(E[X_t]=E[X_{t-1}]=\mu\). Therefore

\[
\mu=c+\phi\mu,
\]

so

\[
\boxed{\mu=\frac{c}{1-\phi}}.
\]

## 3. Variance

After centering around the mean, let \(Y_t=X_t-\mu\). Then

\[
Y_t=\phi Y_{t-1}+\eta_t+\theta\eta_{t-1}.
\]

Squaring and taking expectations gives

\[
\gamma_0=\phi^2\gamma_0+\sigma^2+\theta^2\sigma^2
+2\phi\theta\sigma^2.
\]

The cross-term exists because \(Y_{t-1}\) contains the previous shock \(\eta_{t-1}\).

Therefore

\[
\boxed{\gamma_0=
\frac{\sigma^2(1+\theta^2+2\phi\theta)}{1-\phi^2}}
\]

for \(|\phi|<1\).

## 4. Conditional innovations

Given candidate parameters, the code recursively computes

\[
\hat\eta_t=X_t-c-\phi X_{t-1}-\theta\hat\eta_{t-1}.
\]

The grid-search estimator chooses the \((\phi,\theta)\) pair that minimizes

\[
SSE=\sum_t \hat\eta_t^2.
\]

The intercept is profiled from the sample mean using

\[
c=\bar X(1-\phi).
\]

This provides a deterministic, transparent conditional sum-of-squares formulation that exposes each recurrence step without depending on complex unconstrained optimization routines.

## 5. Forecasting

At time \(T\), the next forecast is

\[
\hat X_{T+1|T}=c+\phi X_T+\theta\hat\eta_T.
\]

For horizons beyond one, the future shocks have conditional expectation zero, so the forecast recursively approaches the stationary mean.
