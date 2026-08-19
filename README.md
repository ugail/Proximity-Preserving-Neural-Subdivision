# Proximity-Preserving Neural Subdivision

*H. Ugail, Proximity-Preserving Neural Subdivision, arXiv:2608.14704, 2026. [https://doi.org/10.21203/rs.3.rs-10502756/v1*](https://arxiv.org/abs/2608.14704)

Classical subdivision is useful because it is repeatable. A single stencil defines the whole refinement rule, the rule can be applied to its own output, and the behaviour of the resulting operator under iteration is understood. That uniformity is also the limitation, because a fixed stencil underfits localised features such as sharp ridges wherever curvature concentrates. Neural mesh refinement adapts to those features, but a network that predicts vertex positions freely is not a subdivision operator, and repeated application drives it out of the classical regime. 

**This is an open-source reference implementation of a trainable subdivision operator that augments Loop subdivision with a small, bounded, curvature-gated correction expressed in a covariant local frame, so that rigid-motion equivariance, affine reproduction, a quadratic proximity envelope around Loop, and planar spectral inheritance all hold by construction, for any finite network weights, before any training takes place.** 

The repository contains the trained checkpoints and every precomputed table, so the complete evaluation reproduces without any training.

For each interior edge with Loop edge vertex $q_i^{0} = \tfrac{3}{8}(p_a + p_b) + \tfrac{1}{8}(p_c + p_d)$, the inserted vertex is

$$q_i^{\theta} \;=\; q_i^{0} \;+\; h_i^{2}\,\gamma_i\,F_i\,\eta_{\theta}(\varphi_i),$$

where $h_i$ is the local edge length, $\gamma_i = \tanh(c\|\varphi_i^{\mathrm{curv}}\|^2) \in [0,1]$ is a curvature gate with a zero of order two on planar input, $F_i \in SO(3)$ is a covariant local frame, and $\eta_\theta$ is a small multilayer perceptron whose output is hard-bounded by $\|\eta_\theta(\varphi_i)\| \le C$.


<img width="3137" height="1455" alt="surface_renders" src="https://github.com/user-attachments/assets/01e73fed-f528-4f51-9561-ee84daba260e" />




The construction is guaranteed by the following proposition:

For any mesh in the shape-regular class, any budget $C$, and any finite network weights, the operator satisfies four properties. Every inserted vertex satisfies $\|q_i^{\theta} - q_i^{0}\| \le C h_i^{2}$, so the operator is $O(h^2)$-proximate to Loop. The operator commutes exactly with every rigid motion. On planar input the gate vanishes and the operator reproduces Loop pointwise. At a planar valence-$k$ star the linearisation of the operator equals Loop's, so it inherits Loop's tangent eigenvalues and Reif spectral gap at that reference configuration. Proofs are in the paper.

## What does this show?

1. **The structural guarantees are architectural, not trained.** Across random weight draws, trained weights, and weights optimised adversarially against each bound, the equivariance residual stays at floating-point noise, the departure from Loop on planar input is exactly zero, and the proximity ratio $\|q_i^\theta - q_i^0\| / (C h_i^2)$ stays at or below one. Gradient ascent that is free in both the network weights and the mesh geometry does not produce a violation.

2. **Loop's spectrum is inherited exactly at the planar reference.** The local subdivision matrix extracted from a planar valence-$k$ star reproduces the analytic tangent eigenvalue $\tfrac{3}{8} + \tfrac{1}{4}\cos(2\pi/k)$ for $k = 3, \ldots, 12$, and the linearisation of the learned operator equals Loop's to machine precision at every one of those valences. Off the planar reference the Reif gap is preserved at zero perturbation for every budget and narrows smoothly in proportion to $C$, which is the sense in which $C$ is a spectral safety parameter as well as a capacity parameter. This is a planar inheritance result and not a global $C^1$ theorem at extraordinary vertices.

3. **Feature approximation improves where fixed stencils underfit.** On ten held-out Gaussian ridges the trained operator improves signed-distance root-mean-square error by 21.0% near the feature, with a bootstrap 95% confidence interval of [13.7, 31.3], and by 13.3% globally with an interval of [7.4, 21.4]. Hausdorff error improves by 9.3% with an interval of [1.7, 21.0]. Normal error is left essentially unchanged, at +0.4% near the feature and +0.1% globally, and neither difference is statistically significant.

4. **One-step accuracy and repeated-application stability are different problems.** The unconstrained foil achieves larger one-step gains, at 56.0% near-feature signed-distance improvement, because free vertex prediction has more capacity to fit a target than gated bounded correction. It also degrades the global normal error to −4.0%, which is the signature of high-frequency artefacts. Through four levels of repeated subdivision its proximity ratio reaches $1.15 \times 10^4$ and its maximum face-normal jump approaches $\pi$, indicating extensive face flips, while the constrained operator stays inside its envelope with the maximum ratio decreasing level by level.

5. **The budget behaves as a real user-facing knob.** Sweeping $C$ over 0.1, 0.25, 0.5 and 1.0 moves the near-feature signed-distance gain from roughly 5% to roughly 22%, with diminishing returns beyond 0.5, and the measured proximity ratio remains at or below the architectural cap at every tested budget. Larger $C$ buys approximation, smaller $C$ buys proximity to Loop and a wider spectral margin.

6. **The gains are feature-driven rather than universal.** On smooth saddle surfaces the operator provides essentially no improvement over Loop, which is the expected behaviour. Saddles are already well approximated by Loop, with truncation error that is small and not localised, so the gate stays close to zero and there is little for a curvature-gated correction to do. The null result is reported as such.

## Contents

- **`run_pipeline_full.ipynb`** — the complete pipeline in a single notebook. Builds the operator with its architectural switches, certifies the bounded-displacement, equivariance and affine-reproduction guarantees under random, trained and adversarial weights, extracts and diagonalises the local subdivision matrix at valence-$k$ stars, ablates each architectural element in turn, trains the operator and the unconstrained foil on the ridge benchmark, runs the paired bootstrap analysis, the repeated-subdivision study, the budget sweep and the capacity control, evaluates the saddle null result and the cross-family sphere-ridge generalisation, and regenerates every figure and metric table.

- **`Results/models/`** — the trained checkpoints. The headline operator at $C = 0.5$, the unconstrained foil, the four budget-sweep operators, the wider capacity-control operator, the saddle operator, and the ablation variants. The notebook loads any checkpoint it finds and skips that training, so the default run on a fresh clone is evaluation-only.

- **`Results/metrics/`** — the precomputed result tables. The held-out ridge comparison is `main_results_table.csv`, the architectural certificates and their adversarial stress tests are `structural_certification.csv`, the valence sweep is `spectral_certification.csv`, the architectural ablation is `ablation.csv`, the budget sweep is `c_sweep.csv`, the cross-family evaluation is `cross_family_sphere_ridge.csv`, and the complete run record including all bootstrap intervals is `metrics_summary.json`. These are sufficient to check every reported number without re-running anything.

- **`Results/figures/`** — the publication figures.

- **`demo/`** — a short standalone notebook that loads the shipped checkpoint and refines a mesh, with the budget exposed as a single parameter.

- **`requirements.txt`** — dependencies (PyTorch, NumPy, Matplotlib).

## Quick start

The fastest way to check the headline numbers is to open the precomputed tables:

```
git clone https://github.com/ugail/Proximity-Preserving-Neural-Subdivision.git
cd Proximity-Preserving-Neural-Subdivision
```

Every value reported in the paper is in `Results/metrics/`, and `metrics_summary.json` contains the complete run record with its bootstrap confidence intervals.

To reproduce the complete evaluation from the shipped checkpoints:

```
pip install -r requirements.txt
jupyter notebook run_pipeline_full.ipynb
```

Run all cells. The notebook detects the included checkpoints, skips all training, and regenerates every figure and table. No GPU is needed for this evaluation-only run. Every path is derived from the repository root, so nothing needs editing before the first run, and the same notebook runs unchanged in Google Colab.

## Reproducing the full pipeline from scratch

The notebook is **incremental**. Checkpoints found in `Results/models` are loaded and their training is skipped, so deleting a checkpoint file retrains exactly that model, and emptying `Results/models` retrains the whole system. Setting `FORCE_RETRAIN = True` in the configuration cell has the same effect without deleting anything. A complete retraining run takes roughly twenty minutes on a laptop CPU, since the meshes are small and the network is a two-layer multilayer perceptron with sixty-four parameters per hidden layer. For a quick functional check before a full run, set `SMOKE_MODE = True` in the configuration cell, which shrinks every split and shortens every training schedule.

## Method variants and metrics

The operator class is defined by four architectural elements, each exposed as a switch so that its contribution can be isolated. The $h_i^2$ factor gives second-order proximity to Loop. The curvature gate suppresses corrections on planar regions and vanishes there together with its first derivative. The covariant frame combined with rigid-motion-invariant features gives exact equivariance. The bounded head enforces $\|\eta_\theta\| \le C$ by a single normalisation rather than by a penalty. The ablation removes each in turn and records which property breaks. The comparison set is Loop as the reference scheme and an unconstrained multilayer perceptron predicting an unbounded vertex offset from raw one-ring coordinates as the foil. The foil is a control that isolates the effect of removing the envelope, not a production method.

Approximation is measured by signed-distance root-mean-square error and Hausdorff distance against the analytic surface, split into a near-feature region and the global region, together with mean and maximum face-normal angular error. Structure is measured by the maximum proximity ratio against the architectural cap, the equivariance residual, the departure from Loop on planar input, the tangent eigenvalues and Reif gap of the local subdivision matrix, and the maximum face-normal jump after repeated refinement. All paired comparisons report bootstrap 95% confidence intervals computed from two thousand resamples.

## Who is this for?

- **Practitioners of geometry processing and subdivision modelling** who want feature-sensitive refinement with a prescribed and checkable proximity envelope around a classical scheme, so that adaptivity does not cost the properties that made the classical scheme trustworthy.

- **Researchers in geometric and constraint-aware deep learning** who want a worked example of output-level structural guarantees on a continuous geometric operator, with the guarantee, its adversarial verification, its spectral certification, and its honest cost accounting in one place.

- **Researchers in subdivision theory** who want a construction in which the proximity hypothesis of Wallner and Dyn is satisfied by parameterisation rather than by assumption, together with the numerical evidence for what happens off the planar reference where no exact statement is available.

## Citation

If you use this implementation or the precomputed results, please cite the paper:
> *H. Ugail, Proximity-Preserving Neural Subdivision, arXiv:2608.14704, 2026. [https://doi.org/10.21203/rs.3.rs-10502756/v1*](https://arxiv.org/abs/2608.14704)


## License

Released under the MIT License. See `LICENSE` for the full text.
