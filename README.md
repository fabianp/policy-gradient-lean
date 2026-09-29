# Policy Gradients in Lean 4

Companion Lean 4 formalizations for the [*Policy Gradients* blog series](https://fa.bianp.net/blog/2026/policy-gradient/) by [Fabian Pedregosa](https://fa.bianp.net).

## Contents

- **[`PolicyGradientVariance.lean`](PolicyGradientVariance.lean)** — Companion to [Policy Gradients Part 1: The REINFORCE Estimator](https://fa.bianp.net/blog/2026/policy-gradient/). Formalizes the $\mathcal{O}(T^3)$ variance upper bound (`PolicyGradient.reinforce_variance_bound_of_rewards`) for the REINFORCE score-function gradient estimator in arbitrary real Hilbert spaces:

  $$\mathrm{Var}\!\left(\left(\sum_{t=0}^{T-1} r_t\right) \sum_{t=0}^{T-1} X_t\right) \le T^3 r_{\max}^2 C$$

  under square-integrable scores $X_t$ with second moment bounded by $C$, zero expected cross inner products ($\mathbb{E}[\langle X_i, X_j \rangle] = 0$ for $i \ne j$), and almost-surely bounded per-step rewards $|r_t| \le r_{\max}$.

## Reproducing & Checking the Proofs

With [Elan](https://github.com/leanprover/elan) installed, clone the repository, fetch the precompiled Mathlib cache, and build:

```bash
git clone https://github.com/fabianp/policy-gradient-lean.git
cd policy-gradient-lean
lake exe cache get Mathlib.MeasureTheory.Function.L2Space Mathlib.Tactic
lake build
```

To step through the tactic states interactively, open this repository's root directory in VS Code with the [Lean 4 extension](https://marketplace.visualstudio.com/items?itemName=leanprover.lean4).
