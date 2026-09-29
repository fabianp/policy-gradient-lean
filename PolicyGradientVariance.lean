import Mathlib.MeasureTheory.Function.L2Space
import Mathlib.Tactic

/-!
# REINFORCE variance bound

Companion to [Policy Gradients Part 1: The REINFORCE Estimator](https://fa.bianp.net/blog/2026/policy-gradient/).

The structure of this file mirrors the proof in the blog post step by step:

* **Preamble (`eq:var_first_step`)**:
  `totalVariance_eq` and `totalVariance_le_secondMoment` expand
  `Var(g) = E[‖g‖²] - ‖E[g]‖² ≤ E[‖g‖²]`.
* **Step 1️⃣ Uncorrelated per-step scores (`eq:zero_score_mean` – `eq:per_step_decomp`)**:
  - `cond_inner_eq_zero_of_zero_mean` and `uncorrelated_scores_of_tower` prove that
    when the action score at step `t'` has zero conditional expectation given the
    trajectory history (`eq:zero_score_mean`), pulling the earlier score `X_t` (`t < t'`)
    out of the inner expectation via the tower property yields `E[⟪X_t, X_{t'}⟫] = 0`
    (`eq:tower_property` and `eq:uncorrelated_scores`).
  - `orthogonal_of_lt` extends `E[⟪X_t, X_{t'}⟫] = 0` from `t < t'` to all distinct pairs `i ≠ j`.
  - `score_sum_secondMoment` proves the Pythagorean identity (`eq:per_step_decomp`):
    `E[‖∑ t, X_t‖²] = ∑ t, E[‖X_t‖²]`.
* **Step 2️⃣ The return `R(τ)²` scales as `O(T²)` (`eq:return_bound`)**:
  `return_abs_bound` and `return_sq_bound` prove `|∑ t, r_t| ≤ T rmax` and
  `(∑ t, r_t)² ≤ T² rmax²` from `|r_t| ≤ rmax`.
* **Step 3️⃣ Summing over all time steps (`eq:variance_split`)**:
  `reinforce_variance_bound` and `reinforce_variance_bound_of_rewards` combine Steps 1️⃣ and 2️⃣
  with the second-moment bound `E[‖X_t‖²] ≤ C` in a `calc` block matching `eq:variance_split`:

      Var((∑ t, r_t) • (∑ t, X_t)) ≤ E[‖R • S‖²]
                                   ≤ T² rmax² E[‖S‖²]
                                   = T² rmax² ∑ t, E[‖X_t‖²]
                                   ≤ T³ rmax² C.

To reproduce the check with Elan installed, run from this directory:

    lake exe cache get Mathlib.MeasureTheory.Function.L2Space Mathlib.Tactic
    lake build

The toolchain and Mathlib revision are pinned. The file ends with `#print axioms`
for the main theorem, so Lean reports the axioms used by the proof.
-/

open MeasureTheory Filter
open scoped BigOperators ENNReal InnerProductSpace

noncomputable section
namespace PolicyGradient

variable {Ω E : Type*} [MeasurableSpace Ω]
  [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]
  {μ : Measure Ω} [IsProbabilityMeasure μ]

/-! ### Preamble: Variance decomposition (`eq:var_first_step`) -/

/-- Total (trace) variance of a vector-valued random variable: `Var(g) = E[‖g - E[g]‖²]`. -/
def totalVariance (μ : Measure Ω) (g : Ω → E) : ℝ :=
  ∫ ω, ‖g ω - ∫ z, g z ∂μ‖ ^ 2 ∂μ

/-- Expand total variance into a second moment minus the squared mean:
`Var(g) = E[‖g‖²] - ‖E[g]‖²`. -/
lemma totalVariance_eq {g : Ω → E} (hg : MemLp g 2 μ) :
    totalVariance μ g = (∫ ω, ‖g ω‖ ^ 2 ∂μ) - ‖∫ ω, g ω ∂μ‖ ^ 2 := by
  have hi : Integrable g μ := hg.integrable (by norm_num)
  have hs := hg.norm.integrable_sq
  have hc := (hi.inner_const (𝕜 := ℝ) (∫ ω, g ω ∂μ)).const_mul (2 : ℝ)
  unfold totalVariance
  simp_rw [norm_sub_sq_real]
  rw [integral_add (hs.sub' hc) (integrable_const _), integral_sub hs hc,
    integral_const_mul]
  have hm : (∫ ω, ⟪g ω, ∫ z, g z ∂μ⟫_ℝ ∂μ) = ‖∫ z, g z ∂μ‖ ^ 2 := by
    calc
      _ = ∫ ω, ⟪∫ z, g z ∂μ, g ω⟫_ℝ ∂μ :=
        integral_congr_ae (Eventually.of_forall fun ω => real_inner_comm _ _)
      _ = _ := by rw [integral_inner hi, real_inner_self_eq_norm_sq]
  rw [hm]
  simp
  ring

/-- Dropping the non-negative squared mean `‖E[g]‖²` upper-bounds variance by the second moment
(Equation `eq:var_first_step` in the blog post). -/
lemma totalVariance_le_secondMoment {g : Ω → E} (hg : MemLp g 2 μ) :
    totalVariance μ g ≤ ∫ ω, ‖g ω‖ ^ 2 ∂μ := by
  rw [totalVariance_eq hg]
  exact sub_le_self _ (sq_nonneg _)

/-! ### Step 1️⃣: Uncorrelated per-step scores (`eq:zero_score_mean` – `eq:per_step_decomp`) -/

/-- Inner expectation in `eq:uncorrelated_scores`: conditioned on the history up to state `s_{t'}`,
the earlier score `x = X_t` (`t < t'`) is fixed and pulls out of the action expectation, so a
zero-mean action score (`eq:zero_score_mean`) makes the conditional inner product vanish. -/
lemma cond_inner_eq_zero_of_zero_mean {A : Type*} [MeasurableSpace A] (π : Measure A)
    (x : E) {Y : A → E} (hY : Integrable Y π) (hzero : (∫ a, Y a ∂π) = 0) :
    (∫ a, ⟪x, Y a⟫_ℝ ∂π) = 0 := by
  rw [integral_inner hY, hzero, inner_zero_right]

/-- Full tower-property derivation of `eq:uncorrelated_scores`: if the expectation of `⟪X_t, X_{t'}⟫`
factors via the law of total expectation (`eq:tower_property`) into an outer expectation over the
trajectory history `h : H` and an inner conditional expectation over the action `a : A` drawn from
the policy `π h`, and the conditional score mean vanishes (`eq:zero_score_mean`), then
`E[⟪X_t, X_{t'}⟫] = 0`. -/
lemma uncorrelated_scores_of_tower {H A : Type*} [MeasurableSpace H] [MeasurableSpace A]
    (ν : Measure H) (π : H → Measure A) (Xt : H → E) (Xt' : H → A → E)
    (hY : ∀ h, Integrable (Xt' h) (π h))
    (hzero : ∀ h, (∫ a, Xt' h a ∂(π h)) = 0) :
    (∫ h, (∫ a, ⟪Xt h, Xt' h a⟫_ℝ ∂(π h)) ∂ν) = 0 := by
  simp_rw [ fun h => cond_inner_eq_zero_of_zero_mean (π h) (Xt h) (hY h) (hzero h),
    integral_zero]

omit [CompleteSpace E] [IsProbabilityMeasure μ] in
/-- Symmetry extends `E[⟪X_t, X_{t'}⟫] = 0` from ordered pairs `t < t'` (`eq:uncorrelated_scores`)
to all distinct step pairs `i ≠ j`. -/
lemma orthogonal_of_lt {T : ℕ} (X : Fin T → Ω → E)
    (hlt : ∀ i j : Fin T, i < j → (∫ ω, ⟪X i ω, X j ω⟫_ℝ ∂μ) = 0) :
    ∀ i j : Fin T, i ≠ j → (∫ ω, ⟪X i ω, X j ω⟫_ℝ ∂μ) = 0 := by
  intro i j hij
  rcases lt_or_gt_of_ne hij with h | h
  · exact hlt i j h
  · calc
      (∫ ω, ⟪X i ω, X j ω⟫_ℝ ∂μ) = ∫ ω, ⟪X j ω, X i ω⟫_ℝ ∂μ :=
        integral_congr_ae (Eventually.of_forall fun ω => real_inner_comm _ _)
      _ = 0 := hlt j i h

omit [CompleteSpace E] [IsProbabilityMeasure μ] in
/-- Equation `eq:per_step_decomp`: uncorrelated score increments (`E[⟪X_i, X_j⟫] = 0` for `i ≠ j`)
make the second moment of the score sum equal the sum of per-step second moments. -/
lemma score_sum_secondMoment {T : ℕ} (X : Fin T → Ω → E)
    (hX : ∀ t, MemLp (X t) 2 μ)
    (horth : ∀ i j, i ≠ j → (∫ ω, ⟪X i ω, X j ω⟫_ℝ ∂μ) = 0) :
    (∫ ω, ‖∑ t, X t ω‖ ^ 2 ∂μ) = ∑ t, ∫ ω, ‖X t ω‖ ^ 2 ∂μ := by
  have hint (i j : Fin T) : Integrable (fun ω => ⟪X i ω, X j ω⟫_ℝ) μ := by
    apply ((hX i).norm.integrable_sq.add (hX j).norm.integrable_sq).mono'
      ((hX i).aestronglyMeasurable.inner (hX j).aestronglyMeasurable)
    filter_upwards [] with ω
    change ‖⟪X i ω, X j ω⟫_ℝ‖ ≤ ‖X i ω‖ ^ 2 + ‖X j ω‖ ^ 2
    have hb := norm_inner_le_norm (𝕜 := ℝ) (X i ω) (X j ω)
    nlinarith [sq_nonneg (‖X i ω‖ - ‖X j ω‖)]
  simp_rw [← real_inner_self_eq_norm_sq, sum_inner, inner_sum]
  rw [integral_finsetSum _ (fun i _ => integrable_finsetSum _ (fun j _ => hint i j))]
  apply Finset.sum_congr rfl
  intro i _
  rw [integral_finsetSum _ (fun j _ => hint i j)]
  exact Finset.sum_eq_single i (fun j _ hji => horth i j hji.symm)
    (by simp)

/-! ### Step 2️⃣: The return `R(τ)²` scales as `O(T²)` (`eq:return_bound`) -/

/-- By the triangle inequality, bounding each per-step reward `|r_t| ≤ rmax` bounds the total
return `|R(τ)| = |∑ t, r_t| ≤ T rmax`. -/
lemma return_abs_bound {T : ℕ} (r : Fin T → ℝ) (rmax : ℝ)
    (hr : ∀ t, |r t| ≤ rmax) : |∑ t, r t| ≤ (T : ℝ) * rmax := by
  calc
    |∑ t, r t| ≤ ∑ t, |r t| := Finset.abs_sum_le_sum_abs _ _
    _ ≤ (T : ℝ) * rmax := by
      simpa using Finset.sum_le_sum (s := Finset.univ) (fun t _ => hr t)

/-- Squaring `return_abs_bound` gives Equation `eq:return_bound`:
`R(τ)² = (∑ t, r_t)² ≤ T² rmax²`. -/
lemma return_sq_bound {T : ℕ} (r : Fin T → ℝ) (rmax : ℝ)
    (hr : ∀ t, |r t| ≤ rmax) : (∑ t, r t) ^ 2 ≤ (T : ℝ) ^ 2 * rmax ^ 2 := by
  have habs := return_abs_bound r rmax hr
  have hpow := pow_le_pow_left₀ (abs_nonneg (∑ t, r t)) habs 2
  nlinarith [sq_abs (∑ t, r t)]

/-! ### Step 3️⃣: Summing over all time steps (`eq:variance_split`) -/

/-- Core analytic estimate corresponding to Equation `eq:variance_split` in the blog post.
The `calc` block mirrors the four steps of `eq:variance_split` line for line. -/
theorem reinforce_variance_bound {T : ℕ} (X : Fin T → Ω → E) (R : Ω → ℝ)
    (rmax C : ℝ) (_hrmax : 0 ≤ rmax)
    (hX : ∀ t, MemLp (X t) 2 μ)
    (horth : ∀ i j, i ≠ j → (∫ ω, ⟪X i ω, X j ω⟫_ℝ ∂μ) = 0)
    (hsecond : ∀ t, (∫ ω, ‖X t ω‖ ^ 2 ∂μ) ≤ C)
    (hR : AEStronglyMeasurable R μ)
    (hbound : ∀ᵐ ω ∂μ, |R ω| ≤ (T : ℝ) * rmax) :
    totalVariance μ (fun ω => R ω • ∑ t, X t ω) ≤ (T : ℝ) ^ 3 * rmax ^ 2 * C := by
  let S : Ω → E := fun ω => ∑ t, X t ω
  have hS : MemLp S 2 μ := memLp_finsetSum _ (fun t _ => hX t)
  have hg : MemLp (fun ω => R ω • S ω) 2 μ := by
    apply hS.of_le_mul (hR.smul hS.aestronglyMeasurable)
    filter_upwards [hbound] with ω hω
    simpa [norm_smul, Real.norm_eq_abs] using
      mul_le_mul_of_nonneg_right hω (norm_nonneg (S ω))
  have hpoint : ∀ᵐ ω ∂μ, ‖R ω • S ω‖ ^ 2 ≤
      ((T : ℝ) * rmax) ^ 2 * ‖S ω‖ ^ 2 := by
    filter_upwards [hbound] with ω hω
    rw [norm_smul, mul_pow, Real.norm_eq_abs]
    exact mul_le_mul_of_nonneg_right
      (pow_le_pow_left₀ (abs_nonneg _) hω 2) (sq_nonneg _)
  calc
    totalVariance μ (fun ω => R ω • ∑ t, X t ω)
        ≤ ∫ ω, ‖R ω • S ω‖ ^ 2 ∂μ :=
          totalVariance_le_secondMoment hg
    _ ≤ ∫ ω, ((T : ℝ) * rmax) ^ 2 * ‖S ω‖ ^ 2 ∂μ :=
      integral_mono_ae hg.norm.integrable_sq
        (hS.norm.integrable_sq.const_mul _) hpoint
    _ = ((T : ℝ) * rmax) ^ 2 * ∑ t, ∫ ω, ‖X t ω‖ ^ 2 ∂μ := by
      rw [integral_const_mul, score_sum_secondMoment X hX horth]
    _ ≤ ((T : ℝ) * rmax) ^ 2 * ((T : ℝ) * C) := by
      apply mul_le_mul_of_nonneg_left _ (sq_nonneg _)
      simpa using Finset.sum_le_sum (s := Finset.univ) (fun t _ => hsecond t)
    _ = (T : ℝ) ^ 3 * rmax ^ 2 * C := by ring

/-- The main theorem stated in the blog post (`eq:variance_bound`), combining Step 1️⃣
(`orthogonal_of_lt` + `score_sum_secondMoment`), Step 2️⃣ (`return_abs_bound`), and Step 3️⃣
(`reinforce_variance_bound`). -/
theorem reinforce_variance_bound_of_rewards {T : ℕ}
    (X : Fin T → Ω → E) (r : Fin T → Ω → ℝ) (rmax C : ℝ)
    (hrmax : 0 ≤ rmax) (hX : ∀ t, MemLp (X t) 2 μ)
    (horth : ∀ i j, i ≠ j → (∫ ω, ⟪X i ω, X j ω⟫_ℝ ∂μ) = 0)
    (hsecond : ∀ t, (∫ ω, ‖X t ω‖ ^ 2 ∂μ) ≤ C)
    (hrmeas : ∀ t, AEStronglyMeasurable (r t) μ)
    (hr : ∀ᵐ ω ∂μ, ∀ t, |r t ω| ≤ rmax) :
    totalVariance μ (fun ω => (∑ t, r t ω) • ∑ t, X t ω)
      ≤ (T : ℝ) ^ 3 * rmax ^ 2 * C := by
  apply reinforce_variance_bound X _ rmax C hrmax hX horth hsecond
    (by fun_prop)
  filter_upwards [hr] with ω hω
  exact return_abs_bound (fun t => r t ω) rmax hω

end PolicyGradient

#print axioms PolicyGradient.reinforce_variance_bound_of_rewards
