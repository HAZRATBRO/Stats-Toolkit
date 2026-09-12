# Stats Toolkit

An interactive, client-side statistics playground built as a single React component. It covers four core topics — probability distributions, expectation & moments, linear regression, and the Central Limit Theorem — with live charts and editable parameters so you can see the math respond in real time.

## Features

### 1. Distribution Explorer
Pick from six standard distributions and adjust their parameters with sliders:
- **Discrete:** Binomial, Poisson, Geometric
- **Continuous:** Normal, Exponential, Uniform

The chart plots the PMF/PDF (with an optional CDF overlay) and a table compares moments computed numerically from the plotted curve against their closed-form formulas (mean, variance, standard deviation, skewness, excess kurtosis).

### 2. Expectation & Moments Calculator
Build a custom discrete distribution by editing an `x` / `P(X=x)` table directly. The app shows the step-by-step calculation (`x·p(x)`, `x²·p(x)`) and derives `E[X]`, `E[X²]`, `Var(X)`, `SD(X)`, skewness, and excess kurtosis live as you type, plus a "normalize to sum = 1" helper.

### 3. 2D Data Generator & Regression
Generate synthetic scatter data from a chosen relationship (linear, quadratic, exponential, logarithmic, sinusoidal, or pure noise) with adjustable coefficients and noise level. An ordinary least squares (OLS) fit line is computed and overlaid, alongside correlation, R², and covariance statistics for both variables.

### 4. Central Limit Theorem Simulator
Repeatedly draw samples of size `n` from a chosen parent distribution (Uniform, Exponential, or a skewed Bernoulli) and plot the distribution of the sample mean against its theoretical normal approximation — demonstrating `Var(X̄) = σ²/n` and the convergence to normality as `n` grows.

## Tech Stack

- **React** (hooks: `useState`, `useMemo`, `useCallback`)
- **Recharts** for all charting (bar, line, composed, and scatter charts)
- **lucide-react** for icons

All statistics are computed from scratch in plain JavaScript, including:
- A log-gamma function (Lanczos approximation) for numerically stable binomial/Poisson PMFs at large `n`/`λ`
- An `erf`-based approximation for the normal CDF
- Box–Muller transform for generating normal random variates
- Manual computation of sample mean, variance, skewness, kurtosis, and covariance

## Usage

`StatsToolkit` is a self-contained default-exported React component (`src/StatsToolkit.jsx`) with its styles inlined via a `<style>` tag — no external CSS file or CSS framework is required.

```jsx
import StatsToolkit from "./src/StatsToolkit";

export default function App() {
  return <StatsToolkit />;
}
```

### Dependencies

```bash
npm install react recharts lucide-react
```

Drop the component into any React project (Vite, Create React App, Next.js, etc.) that supports JSX and has the above packages installed.
