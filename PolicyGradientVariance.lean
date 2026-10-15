import Mathlib.MeasureTheory.Function.L2Space
import Mathlib.Tactic

/-!
# REINFORCE variance bound

Companion to [Policy Gradients Part 1: The REINFORCE Estimator](https://fa.bianp.net/blog/2026/policy-gradient/).

The structure of this file mirrors the definitions and proof in the blog post step by step:

* **Definitions (`eq:reinforce`)**:
  - `trajectoryReturn r τ = ∑ t, r_t(τ)` is the total scalar return `R(τ) ∈ ℝ`.
  - `scoreSum X τ = ∑ t, X_t(τ)` is the sum of per-step score vectors `S(τ) ∈ E`.
  - `reinforceEstimator X r τ = ∑ t, trajectoryReturn r τ • X_t(τ)` is the REINFORCE gradient
    estimator `ĝ(τ)` (where `•` denotes scalar multiplication of a real scalar in `ℝ` by a
    vector in `E`), and `reinforceEstimator_eq_smul_scoreSum` factors the scalar return out of
    the sum: `ĝ(τ) = R(τ) • S(τ)`.
* **Preamble (`eq:var_first_step`)**:
  `totalVariance_eq` and `totalVariance_le_secondMoment` expand
  `Var(g) = E[‖g‖²] - ‖E[g]‖² ≤ E[‖g‖²]`.
* **Step 1️⃣ Uncorrelated per-step scores (`eq:zero_score_mean` – `eq:per_step_decomp`)**:
  - `policy_grad_sum_eq_zero` and `finite_policy_score_zero_mean` differentiate the policy
    normalization identity `∑ a, π_{θ'}(a | h_t) = 1` to prove that the action score has zero
    conditional expectation (`eq:zero_score_mean`).
  - `tower_property` and `orthogonal_scores_of_finite_policy` prove the law of total expectation
    (`eq:tower_property`) and combine it with zero conditional score mean and `orthogonal_of_lt`
    to obtain `E[⟪X_i, X_j⟫] = 0` for all `i ≠ j` (`eq:uncorrelated_scores`).
  - `score_sum_secondMoment` proves the Pythagorean identity (`eq:per_step_decomp`):
    `E[‖scoreSum X‖²] = ∑ t, E[‖X_t‖²]`.
* **Step 2️⃣ The return `R(τ)²` scales as `O(T²)` (`eq:return_bound`)**:
  `return_abs_bound` and `return_sq_bound` prove `|R(τ)| ≤ T rmax` and
  `R(τ)² ≤ T² rmax²` from `|r_t| ≤ rmax`.
* **Step 3️⃣ Summing over all time steps (`eq:variance_split`)**:
  `reinforce_variance_bound` and `reinforce_variance_bound_of_rewards` combine Steps 1️⃣ and 2️⃣
  with the second-moment bound `E[‖X_t‖²] ≤ C` in a `calc` block matching `eq:variance_split`:

      totalVariance μ (reinforceEstimator X r) ≤ E[‖R • S‖²]
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

/-- Lightweight expectation notations wrapping Bochner integration `∫ τ, ... ∂μ`. -/
scoped notation "𝔼[" μ "; " f "]" => ∫ τ, f τ ∂μ
scoped macro "𝔼[" x:ident " ∂" μ:term ", " e:term "]" : term => `(integral $μ (fun $x => $e))

variable {Ω E : Type*} [MeasurableSpace Ω]
  [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]
  {μ : Measure Ω} [IsProbabilityMeasure μ]

/-! ### Definitions: Return, score sum, and REINFORCE estimator (`eq:reinforce`) -/

/-- Total trajectory return `R(τ) = ∑ t, r_t(τ)` (a real scalar). -/
def trajectoryReturn {T : ℕ} (r : Fin T → Ω → ℝ) (τ : Ω) : ℝ :=
  ∑ t, r t τ

/-- Sum of per-step score vectors `S(τ) = ∑ t, X_t(τ)`, where `X_t = ∇_θ log π_θ(a_t | s_t)`. -/
def scoreSum {T : ℕ} (X : Fin T → Ω → E) (τ : Ω) : E :=
  ∑ t, X t τ

/-- The REINFORCE gradient estimator `ĝ(τ) = ∑ t, R(τ) • X_t(τ)` (`eq:reinforce`), written as a sum
of per-step gradient contributions. Here `•` denotes scalar multiplication (`ℝ` acting on `E`). -/
def reinforceEstimator {T : ℕ} (X : Fin T → Ω → E) (r : Fin T → Ω → ℝ) (τ : Ω) : E :=
  ∑ t, trajectoryReturn r τ • X t τ

omit [MeasurableSpace Ω] [CompleteSpace E] in
/-- Factor the scalar return `R(τ)` out of the sum over time steps:
`ĝ(τ) = ∑ t, R(τ) • X_t(τ) = R(τ) • S(τ)`. -/
lemma reinforceEstimator_eq_smul_scoreSum {T : ℕ} (X : Fin T → Ω → E) (r : Fin T → Ω → ℝ) (τ : Ω) :
    reinforceEstimator X r τ = trajectoryReturn r τ • scoreSum X τ := by
  simp [reinforceEstimator, scoreSum, Finset.smul_sum]

/-! ### Preamble: Variance decomposition (`eq:var_first_step`) -/

/-- Total (trace) variance of a vector-valued random variable: `Var(g) = E[‖g - E[g]‖²]`. -/
def totalVariance (μ : Measure Ω) (g : Ω → E) : ℝ :=
  𝔼[τ ∂μ, ‖g τ - 𝔼[μ; g]‖ ^ 2]

/-- Expand total variance into a second moment minus the squared mean:
`Var(g) = E[‖g‖²] - ‖E[g]‖²`. -/
lemma totalVariance_eq {g : Ω → E} (hg : MemLp g 2 μ) :
    totalVariance μ g = 𝔼[τ ∂μ, ‖g τ‖ ^ 2] - ‖𝔼[μ; g]‖ ^ 2 := by
  have hi : Integrable g μ := hg.integrable (by norm_num)
  have hs := hg.norm.integrable_sq
  have hc := (hi.inner_const (𝕜 := ℝ) 𝔼[μ; g]).const_mul (2 : ℝ)
  unfold totalVariance
  simp_rw [norm_sub_sq_real]
  rw [integral_add (hs.sub' hc) (integrable_const _), integral_sub hs hc,
    integral_const_mul]
  have hm : 𝔼[τ ∂μ, ⟪g τ, 𝔼[μ; g]⟫_ℝ] = ‖𝔼[μ; g]‖ ^ 2 := by
    calc
      _ = 𝔼[τ ∂μ, ⟪𝔼[μ; g], g τ⟫_ℝ] :=
        integral_congr_ae (Eventually.of_forall fun τ => real_inner_comm _ _)
      _ = _ := by rw [integral_inner hi, real_inner_self_eq_norm_sq]
  rw [hm]
  simp
  ring

/-- Dropping the non-negative squared mean `‖E[g]‖²` upper-bounds variance by the second moment
(Equation `eq:var_first_step` in the blog post). -/
lemma totalVariance_le_secondMoment {g : Ω → E} (hg : MemLp g 2 μ) :
    totalVariance μ g ≤ 𝔼[τ ∂μ, ‖g τ‖ ^ 2] := by
  rw [totalVariance_eq hg]
  exact sub_le_self _ (sq_nonneg _)

/-! ### Step 1️⃣: Uncorrelated per-step scores (`eq:zero_score_mean` – `eq:per_step_decomp`) -/

omit [MeasurableSpace Ω] [CompleteSpace E] [IsProbabilityMeasure μ] in
/-- `HasGradientAt f g θ` means `g ∈ E` is the gradient `∇_θ f(θ)` in the Hilbert space `E`,
i.e., the Fréchet derivative of `f` at `θ` is the inner-product linear form `v ↦ ⟪g, v⟫_ℝ`. -/
def HasGradientAt (f : E → ℝ) (g : E) (θ : E) : Prop :=
  HasFDerivAt f (innerSL ℝ g) θ

omit [MeasurableSpace Ω] [CompleteSpace E] [IsProbabilityMeasure μ] in
/-- Differentiating the policy normalization identity `∑ a, π_{θ'}(a) = 1` with respect to `θ'`
at `θ` proves that the policy gradients sum to zero: `∑ a, ∇_θ π_θ(a) = 0`. -/
lemma policy_grad_sum_eq_zero {A : Type*} [Fintype A]
    (π : E → A → ℝ) (grad_π : A → E) (θ : E)
    (hnorm : ∀ θ', ∑ a : A, π θ' a = 1)
    (hgrad : ∀ a : A, HasGradientAt (fun θ' => π θ' a) (grad_π a) θ) :
    ∑ a : A, grad_π a = 0 := by
  have hsum : HasFDerivAt (fun θ' => ∑ a : A, π θ' a) (∑ a : A, innerSL ℝ (grad_π a)) θ := by
    simpa using HasFDerivAt.fun_sum (u := Finset.univ) (fun a _ => hgrad a)
  have hconst : HasFDerivAt (fun θ' => ∑ a : A, π θ' a) (0 : E →L[ℝ] ℝ) θ := by
    rw [show (fun θ' => ∑ a : A, π θ' a) = fun _ => (1 : ℝ) from funext hnorm]
    exact hasFDerivAt_const (1 : ℝ) θ
  have heq : ∑ a : A, innerSL ℝ (grad_π a) = 0 := hsum.unique hconst
  rw [← map_sum] at heq
  have hinner : ⟪∑ a : A, grad_π a, ∑ a : A, grad_π a⟫_ℝ = 0 := by
    change innerSL ℝ (∑ a : A, grad_π a) (∑ a : A, grad_π a) = 0
    rw [heq]
    rfl
  exact inner_self_eq_zero.mp hinner

omit [MeasurableSpace Ω] [CompleteSpace E] [IsProbabilityMeasure μ] in
/-- Equation `eq:zero_score_mean` proved from policy normalization `∑ a, π_{θ'}(a) = 1`:
for a differentiable policy `π` with non-zero action probabilities `π θ a ≠ 0`, the expected
log-derivative score `∑ a, π_θ(a) • (π_θ(a)⁻¹ • ∇_θ π_θ(a))` simplifies to `∑ a, ∇_θ π_θ(a)`,
which equals `0` by `policy_grad_sum_eq_zero`. -/
lemma finite_policy_score_zero_mean {A : Type*} [Fintype A]
    (π : E → A → ℝ) (grad_π : A → E) (θ : E)
    (hpos : ∀ a : A, π θ a ≠ 0)
    (hnorm : ∀ θ', ∑ a : A, π θ' a = 1)
    (hgrad : ∀ a : A, HasGradientAt (fun θ' => π θ' a) (grad_π a) θ) :
    ∑ a : A, π θ a • ((π θ a)⁻¹ • grad_π a) = 0 := by
  have hcancel : ∀ a : A, π θ a • ((π θ a)⁻¹ • grad_π a) = grad_π a := fun a => by
    rw [smul_smul, mul_inv_cancel₀ (hpos a), one_smul]
  simp_rw [hcancel]
  exact policy_grad_sum_eq_zero π grad_π θ hnorm hgrad

omit [CompleteSpace E] [IsProbabilityMeasure μ] in
/-- Symmetry extends `E[⟪X_t, X_{t'}⟫] = 0` from ordered pairs `t < t'` (`eq:uncorrelated_scores`)
to all distinct step pairs `i ≠ j`. -/
lemma orthogonal_of_lt {T : ℕ} (X : Fin T → Ω → E)
    (hlt : ∀ i j : Fin T, i < j → 𝔼[τ ∂μ, ⟪X i τ, X j τ⟫_ℝ] = 0) :
    ∀ i j : Fin T, i ≠ j → 𝔼[τ ∂μ, ⟪X i τ, X j τ⟫_ℝ] = 0 := by
  intro i j hij
  rcases lt_or_gt_of_ne hij with h | h
  · exact hlt i j h
  · calc
      𝔼[τ ∂μ, ⟪X i τ, X j τ⟫_ℝ] = 𝔼[τ ∂μ, ⟪X j τ, X i τ⟫_ℝ] :=
        integral_congr_ae (Eventually.of_forall fun τ => real_inner_comm _ _)
      _ = 0 := hlt j i h

omit [MeasurableSpace Ω] [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]
  [IsProbabilityMeasure μ] in
/-- Per-step score vector `X_t(τ) = ∇_θ log π_θ(a_t | h_t) = π_θ(a_t | h_t)⁻¹ • ∇_θ π_θ(a_t | h_t)`
defined via the log-derivative identity (`eq:log_derivative_trick`). -/
def policyScore {T : ℕ} {H A : Type*}
    (π : E → Fin T → H → A → ℝ) (grad_π : Fin T → H → A → E) (θ : E)
    (hist : Fin T → Ω → H) (act : Fin T → Ω → A) (t : Fin T) (τ : Ω) : E :=
  (π θ t (hist t τ) (act t τ))⁻¹ • grad_π t (hist t τ) (act t τ)

omit [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E] in
/-- Law of total expectation (`eq:tower_property`) proved on a finite trajectory space `Ω`:
partitioning trajectories by the history `hist τ = h` and action `act τ = a` via
`Finset.sum_fiberwise` and substituting `ℙ(hist = h, act = a) = ℙ(hist = h) · π_θ(a | h)`
factors the expectation into an outer sum over histories `h` and an inner sum over actions `a`. -/
lemma tower_property [Fintype Ω] [DiscreteMeasurableSpace Ω]
    {H A : Type*} [Fintype H] [DecidableEq H] [Fintype A] [DecidableEq A]
    (hist : Ω → H) (act : Ω → A) (π_step : H → A → ℝ) (f : H → A → ℝ)
    (hpolicy : ∀ h a,
      (∑ τ ∈ Finset.univ.filter (fun τ : Ω => hist τ = h ∧ act τ = a), μ.real {τ}) =
        (∑ τ ∈ Finset.univ.filter (fun τ : Ω => hist τ = h), μ.real {τ}) * π_step h a) :
    𝔼[τ ∂μ, f (hist τ) (act τ)] =
      ∑ h : H, (∑ τ ∈ Finset.univ.filter (fun τ : Ω => hist τ = h), μ.real {τ}) *
        ∑ a : A, π_step h a * f h a := by
  rw [integral_fintype .of_finite]
  simp_rw [smul_eq_mul]
  rw [← Finset.sum_fiberwise Finset.univ (fun τ : Ω => (hist τ, act τ))]
  rw [Fintype.sum_prod_type]
  refine Finset.sum_congr rfl fun h _ => ?_
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun a _ => ?_
  have hfib : (∑ τ ∈ Finset.univ.filter (fun τ : Ω => (hist τ, act τ) = (h, a)),
      μ.real {τ} * f (hist τ) (act τ)) =
      (∑ τ ∈ Finset.univ.filter (fun τ : Ω => hist τ = h ∧ act τ = a), μ.real {τ}) * f h a := by
    rw [Finset.sum_mul]
    refine Finset.sum_congr ?_ fun τ hτ => ?_
    · ext τ
      simp [Prod.ext_iff]
    · simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hτ
      rw [hτ.1, hτ.2]
  rw [hfib, hpolicy h a, mul_assoc]

omit [CompleteSpace E] in
/-- A `T`-step trajectory rollout under a differentiable policy `π_θ` on a finite state/context
space `H` and finite action space `A`, shared between Part 1 (`PolicyGradientVariance`) and Part 2
(`PolicyGradientBaselines`). -/
structure PolicyRollout [Fintype Ω] (μ : Measure Ω) (T : ℕ) (H A : Type*)
    [DecidableEq H] [Fintype A] [DecidableEq A] where
  /-- Conditioning context `h_t(τ) = (s_0, a_0, ..., s_t)` at step `t`. -/
  hist : Fin T → Ω → H
  /-- Action `a_t(τ)` taken at step `t`. -/
  act : Fin T → Ω → A
  /-- Parameterized policy `π_{θ'}(a | h_t)`. -/
  π : E → Fin T → H → A → ℝ
  /-- Policy gradient `∇_θ π_θ(a | h_t)` at parameter `θ`. -/
  grad_π : Fin T → H → A → E
  /-- Current policy parameter vector `θ ∈ E`. -/
  θ : E
  /-- Action probabilities under `π_θ` are non-zero. -/
  hpos : ∀ t h a, π θ t h a ≠ 0
  /-- Policy probabilities sum to 1 (`∑_a π_{θ'}(a | h_t) = 1`) for all `θ'`. -/
  hnorm : ∀ t h θ', ∑ a : A, π θ' t h a = 1
  /-- `grad_π` is the gradient of `θ' ↦ π_{θ'}(a | h_t)` at `θ`. -/
  hgrad : ∀ t h a, HasGradientAt (fun θ' => π θ' t h a) (grad_π t h a) θ
  /-- Action `a_j` is sampled from `π_θ(· | h_j)` given `h_j`:
  `ℙ(h_j = h, a_j = a) = ℙ(h_j = h) · π_θ(a | h_j)`. -/
  hpolicy : ∀ j h a,
    (∑ τ ∈ Finset.univ.filter (fun τ : Ω => hist j τ = h ∧ act j τ = a), μ.real {τ}) =
      (∑ τ ∈ Finset.univ.filter (fun τ : Ω => hist j τ = h), μ.real {τ}) * π θ j h a
  /-- Causality: for `i < j`, the earlier step `(h_i, a_i)` is determined by `h_j`. -/
  hadapt : ∀ i j : Fin T, i < j → ∃ f : H → H × A, ∀ τ, (hist i τ, act i τ) = f (hist j τ)

omit [MeasurableSpace Ω] [CompleteSpace E] [IsProbabilityMeasure μ] in
/-- Per-step score vector `P.score t τ = ∇_θ log π_θ(a_t | h_t)` of a `PolicyRollout`. -/
def PolicyRollout.score [Fintype Ω] {T : ℕ} {H A : Type*} [DecidableEq H] [Fintype A] [DecidableEq A]
    (P : PolicyRollout (E := E) μ T H A) (t : Fin T) (τ : Ω) : E :=
  policyScore P.π P.grad_π P.θ P.hist P.act t τ

omit [MeasurableSpace Ω] [CompleteSpace E] [IsProbabilityMeasure μ] in
/-- Scalar per-step weights `w_t(τ)` are adapted to `P.hist` if for `i < j`, both `w_i(τ)` and
`w_j(τ)` are determined by the conditioning context `P.hist j τ` at step `j`. -/
def PolicyRollout.AdaptedWeight [Fintype Ω] {T : ℕ} {H A : Type*} [DecidableEq H] [Fintype A] [DecidableEq A]
    (P : PolicyRollout (E := E) μ T H A) (w : Fin T → Ω → ℝ) : Prop :=
  ∀ i j : Fin T, i < j → ∃ wi wj : H → ℝ,
    ∀ τ, w i τ = wi (P.hist j τ) ∧ w j τ = wj (P.hist j τ)

omit [CompleteSpace E] in
/-- Shared orthogonality theorem for weighted score increments `w_t(τ) • P.score t τ`:
combining policy normalization (`∑ a, π_{θ'}(a | h_j) = 1`), the log-derivative score identity
(`finite_policy_score_zero_mean`, `eq:zero_score_mean`), the proved tower property
(`tower_property`, `eq:tower_property`), and inner-product symmetry (`orthogonal_of_lt`) proves
that `w_i(τ) • X_i(τ)` and `w_j(τ) • X_j(τ)` are uncorrelated for all `i ≠ j`. -/
lemma PolicyRollout.orthogonal_smul [Fintype Ω] [DiscreteMeasurableSpace Ω]
    {T : ℕ} {H A : Type*} [Fintype H] [DecidableEq H] [Fintype A] [DecidableEq A]
    (P : PolicyRollout (E := E) μ T H A) (w : Fin T → Ω → ℝ)
    (hw : P.AdaptedWeight w) :
    ∀ i j : Fin T, i ≠ j →
      𝔼[τ ∂μ, ⟪w i τ • P.score i τ, w j τ • P.score j τ⟫_ℝ] = 0 := by
  apply orthogonal_of_lt (fun t τ => w t τ • P.score t τ)
  intro i j hij
  rcases P.hadapt i j hij with ⟨f, hf⟩
  rcases hw i j hij with ⟨wi, wj, hwij⟩
  let Yi_of_hj : H → E := fun h =>
    wi h • ((P.π P.θ i (f h).1 (f h).2)⁻¹ • P.grad_π i (f h).1 (f h).2)
  have hYi : ∀ τ, w i τ • P.score i τ = Yi_of_hj (P.hist j τ) := by
    intro τ
    dsimp [PolicyRollout.score, policyScore, Yi_of_hj]
    rw [(hwij τ).1, ← hf τ]
  have hYj : ∀ τ, w j τ • P.score j τ =
      wj (P.hist j τ) • ((P.π P.θ j (P.hist j τ) (P.act j τ))⁻¹ •
        P.grad_π j (P.hist j τ) (P.act j τ)) := by
    intro τ
    dsimp [PolicyRollout.score, policyScore]
    rw [(hwij τ).2]
  have heq : (fun τ => ⟪w i τ • P.score i τ, w j τ • P.score j τ⟫_ℝ) =
      fun τ => ⟪Yi_of_hj (P.hist j τ),
        wj (P.hist j τ) • ((P.π P.θ j (P.hist j τ) (P.act j τ))⁻¹ •
          P.grad_π j (P.hist j τ) (P.act j τ))⟫_ℝ := by
    ext τ
    rw [hYi τ, hYj τ]
  rw [heq, tower_property (P.hist j) (P.act j) (P.π P.θ j)
    (fun h a => ⟪Yi_of_hj h, wj h • ((P.π P.θ j h a)⁻¹ • P.grad_π j h a)⟫_ℝ) (P.hpolicy j)]
  refine Finset.sum_eq_zero fun h _ => ?_
  have hinner_sum : (∑ a : A, P.π P.θ j h a *
      ⟪Yi_of_hj h, wj h • ((P.π P.θ j h a)⁻¹ • P.grad_π j h a)⟫_ℝ) =
      wj h * ⟪Yi_of_hj h, ∑ a : A, P.π P.θ j h a • ((P.π P.θ j h a)⁻¹ • P.grad_π j h a)⟫_ℝ := by
    rw [inner_sum, Finset.mul_sum]
    refine Finset.sum_congr rfl fun a _ => ?_
    simp only [real_inner_smul_right]
    ring
  rw [hinner_sum, finite_policy_score_zero_mean (fun θ' a => P.π θ' j h a) (P.grad_π j h) P.θ
    (P.hpos j h) (P.hnorm j h) (P.hgrad j h), inner_zero_right, mul_zero, mul_zero]

omit [CompleteSpace E] in
/-- Unweighted per-step scores `P.score i` and `P.score j` at distinct time steps `i ≠ j` are
uncorrelated (`eq:uncorrelated_scores`), obtained as the `w ≡ 1` specialization of
`PolicyRollout.orthogonal_smul`. -/
lemma orthogonal_scores_of_finite_policy [Fintype Ω] [DiscreteMeasurableSpace Ω]
    {T : ℕ} {H A : Type*} [Fintype H] [DecidableEq H] [Fintype A] [DecidableEq A]
    (P : PolicyRollout (E := E) μ T H A) :
    ∀ i j : Fin T, i ≠ j → 𝔼[τ ∂μ, ⟪P.score i τ, P.score j τ⟫_ℝ] = 0 := by
  have h1 := P.orthogonal_smul (fun _ _ => (1 : ℝ))
    (fun _ _ _ => ⟨fun _ => 1, fun _ => 1, fun _ => ⟨rfl, rfl⟩⟩)
  simpa [one_smul] using h1

omit [CompleteSpace E] [IsProbabilityMeasure μ] in
/-- Equation `eq:per_step_decomp`: uncorrelated score increments (`E[⟪X_i, X_j⟫] = 0` for `i ≠ j`)
make the second moment of `scoreSum X` equal the sum of per-step second moments. -/
lemma score_sum_secondMoment {T : ℕ} (X : Fin T → Ω → E)
    (hX : ∀ t, MemLp (X t) 2 μ)
    (horth : ∀ i j, i ≠ j → 𝔼[τ ∂μ, ⟪X i τ, X j τ⟫_ℝ] = 0) :
    𝔼[τ ∂μ, ‖scoreSum X τ‖ ^ 2] = ∑ t, 𝔼[τ ∂μ, ‖X t τ‖ ^ 2] := by
  have hint (i j : Fin T) : Integrable (fun τ => ⟪X i τ, X j τ⟫_ℝ) μ := by
    apply ((hX i).norm.integrable_sq.add (hX j).norm.integrable_sq).mono'
      ((hX i).aestronglyMeasurable.inner (hX j).aestronglyMeasurable)
    filter_upwards [] with τ
    change ‖⟪X i τ, X j τ⟫_ℝ‖ ≤ ‖X i τ‖ ^ 2 + ‖X j τ‖ ^ 2
    have hb := norm_inner_le_norm (𝕜 := ℝ) (X i τ) (X j τ)
    nlinarith [sq_nonneg (‖X i τ‖ - ‖X j τ‖)]
  simp only [scoreSum]
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
    (rmax C : ℝ)
    (hX : ∀ t, MemLp (X t) 2 μ)
    (horth : ∀ i j, i ≠ j → 𝔼[τ ∂μ, ⟪X i τ, X j τ⟫_ℝ] = 0)
    (hsecond : ∀ t, 𝔼[τ ∂μ, ‖X t τ‖ ^ 2] ≤ C)
    (hR : AEStronglyMeasurable R μ)
    (hbound : ∀ᵐ τ ∂μ, |R τ| ≤ (T : ℝ) * rmax) :
    totalVariance μ (fun τ => R τ • scoreSum X τ) ≤ (T : ℝ) ^ 3 * rmax ^ 2 * C := by
  -- Define `S(τ) = ∑_{t=0}^{T-1} X_t(τ)` as the sum of per-step score vectors
  let S : Ω → E := scoreSum X
  -- Verify `S ∈ L²(μ)` and `R • S ∈ L²(μ)` so expectations of squared norms are well-defined
  have hS : MemLp S 2 μ := memLp_finsetSum _ (fun t _ => hX t)
  have hg : MemLp (fun τ => R τ • S τ) 2 μ := by
    apply hS.of_le_mul (hR.smul hS.aestronglyMeasurable)
    filter_upwards [hbound] with τ hτ
    simpa [norm_smul, Real.norm_eq_abs] using
      mul_le_mul_of_nonneg_right hτ (norm_nonneg (S τ))
  -- Pointwise Step 2️⃣ bound (`eq:return_bound`): `‖R(τ) S(τ)‖² = R(τ)² ‖S(τ)‖² ≤ T² r_max² ‖S(τ)‖²`
  have hpoint : ∀ᵐ τ ∂μ, ‖R τ • S τ‖ ^ 2 ≤
      ((T : ℝ) * rmax) ^ 2 * ‖S τ‖ ^ 2 := by
    filter_upwards [hbound] with τ hτ
    rw [norm_smul, mul_pow, Real.norm_eq_abs]
    exact mul_le_mul_of_nonneg_right
      (pow_le_pow_left₀ (abs_nonneg _) hτ 2) (sq_nonneg _)
  -- Step 3️⃣ chain of inequalities (`eq:variance_split` in the blog post):
  calc
    -- Line 1 (`eq:var_first_step`): `Var(ĝ) = 𝔼[‖ĝ‖²] - ‖𝔼[ĝ]‖² ≤ 𝔼[‖R(τ) S(τ)‖²]`
    totalVariance μ (fun τ => R τ • scoreSum X τ)
        ≤ 𝔼[τ ∂μ, ‖R τ • S τ‖ ^ 2] :=
          totalVariance_le_secondMoment hg
    -- Line 2 (`eq:return_bound`): bound `R(τ)² ≤ T² r_max²` inside the expectation
    _ ≤ 𝔼[τ ∂μ, ((T : ℝ) * rmax) ^ 2 * ‖S τ‖ ^ 2] :=
      integral_mono_ae hg.norm.integrable_sq
        (hS.norm.integrable_sq.const_mul _) hpoint
    -- Line 3 (`eq:per_step_decomp`): apply the Pythagorean identity `𝔼[‖∑_t X_t‖²] = ∑_t 𝔼[‖X_t‖²]`
    _ = ((T : ℝ) * rmax) ^ 2 * ∑ t, 𝔼[τ ∂μ, ‖X t τ‖ ^ 2] := by
      rw [integral_const_mul, score_sum_secondMoment X hX horth]
    -- Line 4: bound each of the `T` per-step second moments `𝔼[‖X_t‖²] ≤ C`
    _ ≤ ((T : ℝ) * rmax) ^ 2 * ((T : ℝ) * C) := by
      apply mul_le_mul_of_nonneg_left _ (sq_nonneg _)
      simpa using Finset.sum_le_sum (s := Finset.univ) (fun t _ => hsecond t)
    -- Combine factors: `(T r_max)² (T C) = T³ r_max² C`
    _ = (T : ℝ) ^ 3 * rmax ^ 2 * C := by ring

/-- Main theorem (`eq:variance_bound` in the blog post): for a differentiable policy rollout `P`
with per-step score second moments bounded by `C` (`𝔼[‖X_t‖²] ≤ C`) and per-step rewards bounded by
`r_max` (`|r_t(τ)| ≤ r_max`), the total variance of the REINFORCE estimator satisfies
`Var(ĝ) ≤ T³ r_max² C`. -/
theorem reinforce_variance_bound_of_rewards
    [Fintype Ω] [DiscreteMeasurableSpace Ω]
    {T : ℕ} {H A : Type*} [Fintype H] [DecidableEq H] [Fintype A] [DecidableEq A]
    (P : PolicyRollout (E := E) μ T H A)                     -- Differentiable policy rollout over `T` steps
    (r : Fin T → Ω → ℝ)                                      -- Per-step scalar rewards `r_t(τ) ∈ ℝ`
    (rmax C : ℝ)                                             -- Constants `r_max` (reward bound) and `C` (score second-moment bound)
    (hsecond : ∀ t, 𝔼[τ ∂μ, ‖P.score t τ‖ ^ 2] ≤ C)          -- Bounded score second moment (`𝔼[‖X_t‖²] ≤ C`)
    (hr : ∀ τ t, |r t τ| ≤ rmax) :                           -- Bounded per-step rewards (`|r_t(τ)| ≤ r_max` for all `t`)
    totalVariance μ (reinforceEstimator P.score r)           -- Conclusion: total variance `Var(ĝ)` of the REINFORCE estimator
      ≤ (T : ℝ) ^ 3 * rmax ^ 2 * C := by                     -- is bounded above by `T³ r_max² C`
  -- Step 1️⃣ (`eq:zero_score_mean` – `eq:uncorrelated_scores`): prove that scores at distinct
  -- steps `i ≠ j` are uncorrelated (`𝔼[⟪X_i, X_j⟫] = 0`) by differentiating `∑_a π_θ(a | h_j) = 1`
  -- (`finite_policy_score_zero_mean`) and conditioning on `h_j` via the tower property (`tower_property`)
  have horth : ∀ i j, i ≠ j → 𝔼[τ ∂μ, ⟪P.score i τ, P.score j τ⟫_ℝ] = 0 :=
    orthogonal_scores_of_finite_policy P
  -- Factor the scalar return `R(τ) = ∑_t r_t(τ)` out of the sum: `ĝ(τ) = R(τ) • ∑_t X_t(τ)` (`eq:reinforce`)
  rw [show reinforceEstimator P.score r = fun τ => trajectoryReturn r τ • scoreSum P.score τ from
    funext (reinforceEstimator_eq_smul_scoreSum P.score r)]
  -- Step 3️⃣ (`eq:variance_split`): combine Step 1️⃣ (`horth` → Pythagorean identity `eq:per_step_decomp`)
  -- with the score second-moment bound `𝔼[‖X_t‖²] ≤ C` (`hsecond`)
  apply reinforce_variance_bound P.score (trajectoryReturn r) rmax C
    (fun _ => MemLp.of_discrete) horth hsecond AEStronglyMeasurable.of_discrete
  -- Step 2️⃣ (`eq:return_bound`): bound the total return `|R(τ)| = |∑_t r_t(τ)| ≤ T r_max` from `|r_t(τ)| ≤ r_max`
  exact Eventually.of_forall fun τ => return_abs_bound (fun t => r t τ) rmax (hr τ)

end PolicyGradient

#print axioms PolicyGradient.reinforce_variance_bound_of_rewards
