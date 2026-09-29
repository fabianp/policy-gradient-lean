import Mathlib.MeasureTheory.Function.L2Space
import Mathlib.Tactic

/-!
# REINFORCE variance bound

Companion to [Policy Gradients Part 1: The REINFORCE Estimator](https://fa.bianp.net/blog/2026/policy-gradient/).
Scores take values in any real Hilbert space; trajectories live on an arbitrary probability space. The policy-specific
input is orthogonality in expectation (normally proved from the martingale
difference property of scores). No independence of rewards and scores is assumed.

The main theorem, `PolicyGradient.reinforce_variance_bound_of_rewards`, proves

    Var((∑ t, rₜ) • (∑ t, Xₜ)) ≤ T³ rmax² C.

Assumptions and scope:
* Scores take values in any complete real inner product space (including ℝᵈ).
* Each score is in L² and has second moment at most C.
* Distinct scores have zero expected inner product.
* Per-step rewards are measurable and bounded in absolute value by rmax almost
  surely, with rmax ≥ 0.

The horizon can be zero. Neither independence between rewards and scores nor
zero mean of the estimator is required. The policy-specific derivation of score
orthogonality from conditional mean zero is outside this formalization, as are
the score-function differentiation identity and any Θ(T³) lower bound.

The proof establishes the inequalities in this order, with S = ∑ t, Xₜ:

    Var(R • S) ≤ E[‖R • S‖²]
               ≤ (T rmax)² E[‖S‖²]
               = (T rmax)² ∑ t, E[‖Xₜ‖²]
               ≤ T³ rmax² C.

To reproduce the check with Elan installed, run from this directory:

    lake update
    lake exe cache get Mathlib/MeasureTheory/Function/L2Space.lean Mathlib/Tactic.lean
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

/-- Total (trace) variance of a vector-valued random variable. -/
def totalVariance (μ : Measure Ω) (g : Ω → E) : ℝ :=
  ∫ ω, ‖g ω - ∫ z, g z ∂μ‖ ^ 2 ∂μ

/-- Expand total variance into a second moment minus the squared mean. -/
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

omit [CompleteSpace E] [IsProbabilityMeasure μ] in
/-- Orthogonal score increments make the second moment of their sum additive.

The hypothesis `horth` is the formal version of the zero cross-term property
used for policy-gradient scores. -/
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

/-- Bound the variance of `R • ∑ t, X t` when the total return is bounded.

This is the core analytic estimate. The reward structure is intentionally
abstracted away: only measurability and the almost-everywhere bound on `R` are
needed here. -/
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
        ≤ ∫ ω, ‖R ω • S ω‖ ^ 2 ∂μ := by
          rw [totalVariance_eq hg]
          exact sub_le_self _ (sq_nonneg _)
    _ ≤ ∫ ω, ((T : ℝ) * rmax) ^ 2 * ‖S ω‖ ^ 2 ∂μ :=
      integral_mono_ae hg.norm.integrable_sq
        (hS.norm.integrable_sq.const_mul _) hpoint
    _ = ((T : ℝ) * rmax) ^ 2 * ∑ t, ∫ ω, ‖X t ω‖ ^ 2 ∂μ := by
      rw [integral_const_mul, score_sum_secondMoment X hX horth]
    _ ≤ ((T : ℝ) * rmax) ^ 2 * ((T : ℝ) * C) := by
      apply mul_le_mul_of_nonneg_left _ (sq_nonneg _)
      simpa using Finset.sum_le_sum (s := Finset.univ) (fun t _ => hsecond t)
    _ = (T : ℝ) ^ 3 * rmax ^ 2 * C := by ring

/-- The triangle inequality turns per-step reward bounds into a return bound. -/
lemma return_abs_bound {T : ℕ} (r : Fin T → ℝ) (rmax : ℝ)
    (hr : ∀ t, |r t| ≤ rmax) : |∑ t, r t| ≤ (T : ℝ) * rmax := by
  calc
    |∑ t, r t| ≤ ∑ t, |r t| := Finset.abs_sum_le_sum_abs _ _
    _ ≤ (T : ℝ) * rmax := by
      simpa using Finset.sum_le_sum (s := Finset.univ) (fun t _ => hr t)

/-- The theorem stated in the article, with the return written as a reward sum.

This combines `return_abs_bound` with `reinforce_variance_bound`. -/
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
