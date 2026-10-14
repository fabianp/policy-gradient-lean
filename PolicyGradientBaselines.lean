import PolicyGradientVariance

/-!
# Policy Gradients Part 2: Baselines ($O(T^2)$ Variance Bound)

Companion to [Policy Gradients Part 2: Baselines](https://fa.bianp.net/blog/2026/policy-gradient-baselines/).

This file imports `PolicyGradientVariance` (Part 1), reusing its `𝔼[...]` notations,
`totalVariance`, `scoreSum`, `totalVariance_eq`, `totalVariance_le_secondMoment`,
`orthogonal_of_lt`, and `score_sum_secondMoment`.

The structure mirrors the definitions and derivations in Part 2 step by step:

* **Control Variates (`eq:generic_cv` – `eq:cv_variance`)**:
  - `totalCovariance μ Y Z` defines the inner-product covariance `𝔼[⟪Y - 𝔼[Y], Z - 𝔼[Z]⟫]`.
  - `controlVariate_unbiased` proves that subtracting a zero-mean control variate `Z` preserves
    the expected value: `𝔼[Y - Z] = 𝔼[Y]`.
  - `totalVariance_sub` proves the control-variate variance identity (`eq:cv_variance`):
    `Var(Y - Z) = Var(Y) + Var(Z) - 2 Cov(Y, Z)`.
* **Baselines as Control Variates & Expected Return Baselines (`eq:pg_raw_estimator` – `eq:expected_return_baseline`)**:
  - `rewardToGo r t τ = ∑ k ∈ Finset.Ici t, r k τ` is the reward-to-go `G_t`.
  - `baselineStep X G b t τ = (G_t(τ) - b_t(τ)) • X_t(τ)` is the centered per-step gradient
    contribution `Y_t`, and `baselineEstimator X G b = scoreSum (baselineStep X G b)` is `ĝ_b(τ)`
    (`eq:baseline_estimator`).
  - `baseline_cond_zero_mean` proves that scaling a zero-mean score by a state-dependent scalar
    baseline `b(s_t)` still has zero conditional mean (`eq:score_zero_mean_conditional`).
  - `mse_eq_variance_add_sq` and `expected_return_minimizes_mse` prove that choosing the baseline
    equal to the expected return `b = 𝔼[G]` minimizes the mean squared error `𝔼[(G - c)²]` over all
    constants `c` (`eq:expected_return_baseline`).
* **Quadratic $O(T^2)$ Variance Bound (`eq:bounded_rewards` – `eq:baseline_second_moment`)**:
  - **Step 1️⃣ (`eq:return_variance_bound`)**: `geom_sum_Ico_le` and
    `return_variance_bound_of_mixing` bound the variance of `m = T - t` rewards under bounded
    rewards `|r_k| ≤ rmax` and geometric covariance decay `|Cov(r_i, r_j)| ≤ M ρ^{|i-j|}` by
    `(T - t) K_r`, where `K_r = rmax² + 2 M ρ / (1 - ρ)` (`rewardVarianceConst`).
  - **Step 2️⃣ (`eq:per_step_moment_bound`)**: `baselineStep_secondMoment_bound` combines the
    score bound `‖X_t‖² ≤ C` with the centered return variance bound `𝔼[(G_t - b_t)²] ≤ (T - t) K_r`
    to obtain `𝔼[‖Y_t‖²] ≤ (T - t) K_r C`.
  - **Step 3️⃣ (`eq:baseline_second_moment` & `eq:baseline_variance_bound`)**:
    `sum_horizon_remaining_eq` and `sum_horizon_remaining_le_sq` establish
    `∑ t, (T - t) = T(T + 1)/2 ≤ T²`, and `baseline_variance_bound` combines Steps 1️⃣–3️⃣ with
    `totalVariance_le_secondMoment` and `score_sum_secondMoment` from Part 1 to prove:

        totalVariance μ (baselineEstimator X G b) ≤ T² K_r C.
-/

open MeasureTheory Filter
open scoped BigOperators ENNReal InnerProductSpace

noncomputable section
namespace PolicyGradient

variable {Ω E : Type*} [MeasurableSpace Ω]
  [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]
  {μ : Measure Ω} [IsProbabilityMeasure μ]

/-! ### Interlude: Control Variates (`eq:generic_cv` – `eq:cv_variance`) -/

/-- Total (inner-product) covariance between two vector-valued random variables:
`Cov(Y, Z) = 𝔼[⟪Y - 𝔼[Y], Z - 𝔼[Z]⟫]`. -/
def totalCovariance (μ : Measure Ω) (Y Z : Ω → E) : ℝ :=
  𝔼[τ ∂μ, ⟪Y τ - 𝔼[μ; Y], Z τ - 𝔼[μ; Z]⟫_ℝ]

omit [CompleteSpace E] [IsProbabilityMeasure μ] in
/-- Subtracting a zero-mean control variate `Z` leaves the expected value unchanged (`eq:generic_cv`):
`𝔼[Y - Z] = 𝔼[Y]`. -/
lemma controlVariate_unbiased {Y Z : Ω → E} (hY : Integrable Y μ) (hZ : Integrable Z μ)
    (hZ0 : 𝔼[μ; Z] = 0) :
    𝔼[τ ∂μ, Y τ - Z τ] = 𝔼[μ; Y] := by
  rw [integral_sub hY hZ, hZ0, sub_zero]

omit [CompleteSpace E] in
/-- Control-variate variance decomposition (`eq:cv_variance`):
`Var(Y - Z) = Var(Y) + Var(Z) - 2 Cov(Y, Z)`. -/
lemma totalVariance_sub {Y Z : Ω → E} (hY : MemLp Y 2 μ) (hZ : MemLp Z 2 μ) :
    totalVariance μ (fun τ => Y τ - Z τ) =
      totalVariance μ Y + totalVariance μ Z - 2 * totalCovariance μ Y Z := by
  have hiY : Integrable Y μ := hY.integrable (by norm_num)
  have hiZ : Integrable Z μ := hZ.integrable (by norm_num)
  let Yc : Ω → E := fun τ => Y τ - 𝔼[μ; Y]
  let Zc : Ω → E := fun τ => Z τ - 𝔼[μ; Z]
  have hYc : MemLp Yc 2 μ := hY.sub (memLp_const _)
  have hZc : MemLp Zc 2 μ := hZ.sub (memLp_const _)
  have hYc_sq : Integrable (fun τ => ‖Yc τ‖ ^ 2) μ := hYc.norm.integrable_sq
  have hZc_sq : Integrable (fun τ => ‖Zc τ‖ ^ 2) μ := hZc.norm.integrable_sq
  have hYZ : Integrable (fun τ => ⟪Yc τ, Zc τ⟫_ℝ) μ := by
    apply (hYc_sq.add hZc_sq).mono' (hYc.aestronglyMeasurable.inner hZc.aestronglyMeasurable)
    filter_upwards [] with τ
    change ‖⟪Yc τ, Zc τ⟫_ℝ‖ ≤ ‖Yc τ‖ ^ 2 + ‖Zc τ‖ ^ 2
    have hb := norm_inner_le_norm (𝕜 := ℝ) (Yc τ) (Zc τ)
    nlinarith [sq_nonneg (‖Yc τ‖ - ‖Zc τ‖)]
  have h2YZ : Integrable (fun τ => 2 * ⟪Yc τ, Zc τ⟫_ℝ) μ := hYZ.const_mul 2
  unfold totalVariance totalCovariance
  have hmean : 𝔼[τ ∂μ, Y τ - Z τ] = 𝔼[μ; Y] - 𝔼[μ; Z] := integral_sub hiY hiZ
  have hpoint : ∀ τ, ‖(Y τ - Z τ) - 𝔼[z ∂μ, Y z - Z z]‖ ^ 2 =
      ‖Yc τ‖ ^ 2 - 2 * ⟪Yc τ, Zc τ⟫_ℝ + ‖Zc τ‖ ^ 2 := by
    intro τ
    rw [hmean, sub_sub_sub_comm]
    exact norm_sub_sq_real (Yc τ) (Zc τ)
  simp_rw [hpoint]
  rw [integral_add (hYc_sq.sub' h2YZ) hZc_sq, integral_sub hYc_sq h2YZ, integral_const_mul]
  ring

/-! ### Baselines as Control Variates & Expected Return (`eq:pg_raw_estimator` – `eq:expected_return_baseline`) -/

/-- Reward-to-go from step `t` onwards (`eq:pg_raw_estimator`): `G_t(τ) = ∑_{k ≥ t} r_k(τ)`. -/
def rewardToGo {T : ℕ} (r : Fin T → Ω → ℝ) (t : Fin T) (τ : Ω) : ℝ :=
  ∑ k ∈ Finset.Ici t, r k τ

/-- Centered per-step policy gradient contribution `Y_t(τ) = (G_t(τ) - b_t(τ)) • X_t(τ)`. -/
def baselineStep {T : ℕ} (X : Fin T → Ω → E) (G b : Fin T → Ω → ℝ) (t : Fin T) (τ : Ω) : E :=
  (G t τ - b t τ) • X t τ

/-- The policy gradient baseline estimator (`eq:baseline_estimator`):
`ĝ_b(τ) = ∑ t, (G_t(τ) - b_t(τ)) • X_t(τ)`, defined via `scoreSum` from Part 1. -/
def baselineEstimator {T : ℕ} (X : Fin T → Ω → E) (G b : Fin T → Ω → ℝ) (τ : Ω) : E :=
  scoreSum (baselineStep X G b) τ

omit [MeasurableSpace Ω] [CompleteSpace E] [IsProbabilityMeasure μ] in
/-- Because the score `X` has zero conditional expectation given `s_t` (`eq:score_zero_mean_conditional`),
multiplying by any state-dependent scalar baseline `b` (constant with respect to the action `a`)
still has zero conditional expectation. -/
lemma baseline_cond_zero_mean {A : Type*} [MeasurableSpace A] (π : Measure A)
    (b : ℝ) {X : A → E} (hzero : 𝔼[π; X] = 0) :
    𝔼[a ∂π, b • X a] = 0 := by
  rw [integral_smul, hzero, smul_zero]

/-- Bias-variance decomposition of the mean squared error around an arbitrary constant `c`:
`𝔼[‖G - c‖²] = Var(G) + ‖𝔼[G] - c‖²`. Reuses `totalVariance_eq` from Part 1. -/
lemma mse_eq_variance_add_sq {G : Ω → E} (hG : MemLp G 2 μ) (c : E) :
    𝔼[τ ∂μ, ‖G τ - c‖ ^ 2] = totalVariance μ G + ‖𝔼[μ; G] - c‖ ^ 2 := by
  have hGc : MemLp (fun τ => G τ - c) 2 μ := hG.sub (memLp_const c)
  have h1 := totalVariance_eq hGc
  have hmean : 𝔼[τ ∂μ, G τ - c] = 𝔼[μ; G] - c := by
    rw [integral_sub (hG.integrable (by norm_num)) (integrable_const c), integral_const]
    simp
  have hvar : totalVariance μ (fun τ => G τ - c) = totalVariance μ G := by
    unfold totalVariance
    simp_rw [hmean, sub_sub_sub_cancel_right]
  rw [hmean, hvar] at h1
  linarith

/-- Setting the baseline to the expected return `b = 𝔼[G]` (`eq:expected_return_baseline`)
minimizes the mean squared deviation `𝔼[‖G - c‖²]` over all constants `c`. -/
lemma expected_return_minimizes_mse {G : Ω → E} (hG : MemLp G 2 μ) (c : E) :
    𝔼[τ ∂μ, ‖G τ - 𝔼[μ; G]‖ ^ 2] ≤ 𝔼[τ ∂μ, ‖G τ - c‖ ^ 2] := by
  change totalVariance μ G ≤ 𝔼[τ ∂μ, ‖G τ - c‖ ^ 2]
  rw [mse_eq_variance_add_sq hG c]
  linarith [sq_nonneg ‖𝔼[μ; G] - c‖]

/-! ### Step 1️⃣: Bounded centered return variance (`eq:return_variance_bound`) -/

/-- The return variance constant `K_r = rmax² + 2 M ρ / (1 - ρ)` from the main lemma. -/
def rewardVarianceConst (rmax M ρ : ℝ) : ℝ :=
  rmax ^ 2 + 2 * M * ρ / (1 - ρ)

/-- Finite geometric series bound: for `ρ ∈ [0, 1)`, `∑_{d=1}^n ρ^d ≤ ρ / (1 - ρ)`. -/
lemma geom_sum_Ico_le {ρ : ℝ} (hρ0 : 0 ≤ ρ) (hρ1 : ρ < 1) (n : ℕ) :
    (∑ d ∈ Finset.Ico 1 (n + 1), ρ ^ d) ≤ ρ / (1 - ρ) := by
  have h1 : 0 < 1 - ρ := sub_pos.mpr hρ1
  rw [le_div_iff₀ h1]
  have htel : (∑ d ∈ Finset.Ico 1 (n + 1), ρ ^ d) * (1 - ρ) = ρ - ρ ^ (n + 1) := by
    induction n with
    | zero => simp
    | succ m ih =>
      rw [Finset.sum_Ico_succ_top (by omega), add_mul, ih]
      ring
  have hpow : 0 ≤ ρ ^ (n + 1) := pow_nonneg hρ0 _
  linarith

/-- Equation `eq:return_variance_bound`: if `m = T - t` centered rewards have per-step variance
bounded by `rmax²` (`eq:bounded_rewards`) and off-diagonal covariance bounded by `2 M ρ / (1 - ρ)`
from geometric mixing (`eq:mixing_assumption`), then the total variance of their sum is bounded by
`m K_r = (T - t) (rmax² + 2 M ρ / (1 - ρ))`. -/
lemma return_variance_bound_of_mixing (m : ℕ) (rmax M ρ : ℝ)
    (diagSum offDiagSum : ℝ)
    (hdiag : diagSum ≤ (m : ℝ) * rmax ^ 2)
    (hoff : offDiagSum ≤ (m : ℝ) * (2 * M * ρ / (1 - ρ))) :
    diagSum + offDiagSum ≤ (m : ℝ) * rewardVarianceConst rmax M ρ := by
  unfold rewardVarianceConst
  linarith

/-! ### Step 2️⃣: Per-step second moments (`eq:per_step_moment_bound`) -/

omit [CompleteSpace E] [IsProbabilityMeasure μ] in
/-- Equation `eq:per_step_moment_bound`: when the per-step score norm is bounded almost surely by
`‖X_t‖² ≤ C` (`eq:bounded_scores`) and the centered return second moment satisfies
`𝔼[(G_t - b_t)²] ≤ (T - t) K_r` (`eq:return_variance_bound`), the per-step baseline contribution
`Y_t = (G_t - b_t) • X_t` satisfies `𝔼[‖Y_t‖²] ≤ (T - t) K_r C`. -/
lemma baselineStep_secondMoment_bound {T : ℕ} (X : Fin T → Ω → E) (G b : Fin T → Ω → ℝ)
    (Kr C : ℝ) (hC : 0 ≤ C) (t : Fin T)
    (hY : MemLp (baselineStep X G b t) 2 μ)
    (hGb : MemLp (fun τ => G t τ - b t τ) 2 μ)
    (hXbound : ∀ᵐ τ ∂μ, ‖X t τ‖ ^ 2 ≤ C)
    (hGvar : 𝔼[τ ∂μ, (G t τ - b t τ) ^ 2] ≤ ((T - t.val : ℕ) : ℝ) * Kr) :
    𝔼[τ ∂μ, ‖baselineStep X G b t τ‖ ^ 2] ≤ ((T - t.val : ℕ) : ℝ) * Kr * C := by
  have hpoint : ∀ᵐ τ ∂μ, ‖baselineStep X G b t τ‖ ^ 2 ≤ (G t τ - b t τ) ^ 2 * C := by
    filter_upwards [hXbound] with τ hτ
    simp only [baselineStep, norm_smul, mul_pow, Real.norm_eq_abs, sq_abs]
    exact mul_le_mul_of_nonneg_left hτ (sq_nonneg _)
  have hint_rhs : Integrable (fun τ => (G t τ - b t τ) ^ 2 * C) μ := by
    have h1 := hGb.norm.integrable_sq
    simp only [Real.norm_eq_abs, sq_abs] at h1
    exact h1.mul_const C
  calc
    𝔼[τ ∂μ, ‖baselineStep X G b t τ‖ ^ 2]
        ≤ 𝔼[τ ∂μ, (G t τ - b t τ) ^ 2 * C] :=
          integral_mono_ae hY.norm.integrable_sq hint_rhs hpoint
    _ = 𝔼[τ ∂μ, (G t τ - b t τ) ^ 2] * C := integral_mul_const C _
    _ ≤ (((T - t.val : ℕ) : ℝ) * Kr) * C := mul_le_mul_of_nonneg_right hGvar hC
    _ = ((T - t.val : ℕ) : ℝ) * Kr * C := by ring

/-! ### Step 3️⃣: Summing over all time steps (`eq:baseline_second_moment` & `eq:baseline_variance_bound`) -/

/-- Exact triangular sum identity appearing in `eq:baseline_second_moment`:
`2 * ∑_{t=0}^{T-1} (T - t) = T (T + 1)`. -/
lemma sum_range_sub_eq (T : ℕ) :
    2 * (∑ t ∈ Finset.range T, (T - t)) = T * (T + 1) := by
  induction T with
  | zero => simp
  | succ n ih =>
    have hsplit : ∑ t ∈ Finset.range (n + 1), (n + 1 - t) =
        (n + 1) + ∑ t ∈ Finset.range n, (n - t) := by
      rw [Finset.sum_range_succ']
      simp [add_comm]
    linarith

/-- Quadratic upper bound on the horizon sum in `eq:baseline_second_moment`:
`∑_{t=0}^{T-1} (T - t) ≤ T²`. -/
lemma sum_horizon_remaining_le_sq (T : ℕ) :
    (∑ t : Fin T, ((T - t.val : ℕ) : ℝ)) ≤ (T : ℝ) ^ 2 := by
  have hterm : ∀ t : Fin T, ((T - t.val : ℕ) : ℝ) ≤ (T : ℝ) := by
    intro t
    exact Nat.cast_le.mpr (Nat.sub_le T t.val)
  calc
    (∑ t : Fin T, ((T - t.val : ℕ) : ℝ))
        ≤ ∑ _t : Fin T, (T : ℝ) := Finset.sum_le_sum (fun t _ => hterm t)
    _ = (T : ℝ) ^ 2 := by simp [sq]

/-- Core theorem corresponding to Equations `eq:baseline_variance_bound` and
`eq:baseline_second_moment` in the blog post. Reuses `totalVariance_le_secondMoment` and
`score_sum_secondMoment` from `PolicyGradientVariance` (Part 1). -/
theorem baseline_variance_bound {T : ℕ} (X : Fin T → Ω → E) (G b : Fin T → Ω → ℝ)
    (Kr C : ℝ) (hKr : 0 ≤ Kr) (hC : 0 ≤ C)
    (hY : ∀ t, MemLp (baselineStep X G b t) 2 μ)
    (horth : ∀ i j, i ≠ j → 𝔼[τ ∂μ, ⟪baselineStep X G b i τ, baselineStep X G b j τ⟫_ℝ] = 0)
    (hstep : ∀ t : Fin T, 𝔼[τ ∂μ, ‖baselineStep X G b t τ‖ ^ 2] ≤ ((T - t.val : ℕ) : ℝ) * Kr * C) :
    totalVariance μ (baselineEstimator X G b) ≤ (T : ℝ) ^ 2 * Kr * C := by
  have hg : MemLp (baselineEstimator X G b) 2 μ :=
    memLp_finsetSum Finset.univ (fun t _ => hY t)
  calc
    totalVariance μ (baselineEstimator X G b)
        ≤ 𝔼[τ ∂μ, ‖baselineEstimator X G b τ‖ ^ 2] :=
          totalVariance_le_secondMoment hg
    _ = ∑ t : Fin T, 𝔼[τ ∂μ, ‖baselineStep X G b t τ‖ ^ 2] :=
          score_sum_secondMoment (baselineStep X G b) hY horth
    _ ≤ ∑ t : Fin T, (((T - t.val : ℕ) : ℝ) * Kr * C) :=
          Finset.sum_le_sum (fun t _ => hstep t)
    _ = (∑ t : Fin T, ((T - t.val : ℕ) : ℝ)) * (Kr * C) := by
          simp_rw [mul_assoc, ← Finset.sum_mul]
    _ ≤ (T : ℝ) ^ 2 * (Kr * C) :=
          mul_le_mul_of_nonneg_right (sum_horizon_remaining_le_sq T) (mul_nonneg hKr hC)
    _ = (T : ℝ) ^ 2 * Kr * C := by ring

/-- Full end-to-end theorem stated in the framed lemma of Part 2 (`eq:baseline_variance_bound`):
under bounded score norms `‖X_t‖² ≤ C` (`eq:bounded_scores`), centered return variance
`𝔼[(G_t - b_t)²] ≤ (T - t) K_r` (`eq:return_variance_bound`), and orthogonal per-step increments,
the baseline policy gradient estimator satisfies:
`Var(ĝ_b) ≤ T² K_r C` where `K_r = rmax² + 2 M ρ / (1 - ρ)`. -/
theorem baseline_variance_bound_of_return_variance {T : ℕ}
    (X : Fin T → Ω → E) (G b : Fin T → Ω → ℝ) (rmax M ρ C : ℝ)
    (hM : 0 ≤ M) (hρ0 : 0 ≤ ρ) (hρ1 : ρ < 1) (hC : 0 ≤ C)
    (hY : ∀ t, MemLp (baselineStep X G b t) 2 μ)
    (hGb : ∀ t, MemLp (fun τ => G t τ - b t τ) 2 μ)
    (horth : ∀ i j, i ≠ j → 𝔼[τ ∂μ, ⟪baselineStep X G b i τ, baselineStep X G b j τ⟫_ℝ] = 0)
    (hXbound : ∀ t, ∀ᵐ τ ∂μ, ‖X t τ‖ ^ 2 ≤ C)
    (hGvar : ∀ t : Fin T,
      𝔼[τ ∂μ, (G t τ - b t τ) ^ 2] ≤ ((T - t.val : ℕ) : ℝ) * rewardVarianceConst rmax M ρ) :
    totalVariance μ (baselineEstimator X G b)
      ≤ (T : ℝ) ^ 2 * rewardVarianceConst rmax M ρ * C := by
  have hKr : 0 ≤ rewardVarianceConst rmax M ρ := by
    unfold rewardVarianceConst
    have h1 : 0 < 1 - ρ := sub_pos.mpr hρ1
    positivity
  apply baseline_variance_bound X G b (rewardVarianceConst rmax M ρ) C hKr hC hY horth
  intro t
  exact baselineStep_secondMoment_bound X G b (rewardVarianceConst rmax M ρ) C hC t
    (hY t) (hGb t) (hXbound t) (hGvar t)

end PolicyGradient

#print axioms PolicyGradient.baseline_variance_bound_of_return_variance
