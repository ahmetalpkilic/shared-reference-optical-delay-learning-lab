# Fixed calibration and optical-delay batches

A 45–60 minute integrated learning lab: why can a detector average near 5% false alarms yet behave very differently for individual instrument calibrations? This follows the finite common-whitening lesson, replacing fresh calibration on every event with one calibration held across an entire batch. It adds conditional threshold training and a between-calibration variance experiment.

## Learn and run

Prerequisites: Gaussian PSFs, covariance, least squares, conditional probability and basic Python. Objectives: derive pixel-integrated spot localization; convert position to optical delay; distinguish marginal and conditional alarm calibration; quantify batch variability without treating dependent alarms as independent.

```sh
git clone https://github.com/ahmetalpkilic/fixed-calibration-optical-delay-learning-lab.git
cd fixed-calibration-optical-delay-learning-lab
sh run_demo.sh
```

Python 3.9–3.12. Initial installation requires internet; the synthetic demo then runs offline with pinned dependencies. The command creates `.venv`, runs tests and writes `examples/results.json` and `examples/batches.png`. Seed 20261021; cross-platform last-bit differences are possible. No external dataset or weights. Suggested hour: 10 minutes predict, 10 run/inspect, 15 derivation, 15 code, 10 an exercise.

## Four-way connection

| Area | Explicit model | Measurement connection |
|---|---|---|
| Optics | 550 nm Gaussian pupil, amplitude radius 1 mm, focal length 16 mm | Fourier transform determines spot width |
| Camera | 4 µm pixels, exact pixel integration, read-noise SD 2 signal units | Noisy samples determine subpixel position |
| Vision/metrology | Bounded nonlinear spot fit; known 0.8 px/s motion; two timestamp channels | Optical time minus fused clock time estimates a 120 ms delay |
| Mathematics | Common whitening, conditional ranks, total variance, finite-repeat GLS | Fixed covariance changes alarm rates and delay risk across batches |

## Physical model and conventions


Use an untruncated paraxial Gaussian pupil amplitude `P(r)=exp(-r²/a²)`. Fourier transformation and intensity squaring yield

`σ_sensor = λ f/(2πa)`, `σ_px = λ f/(2πap) ≈ 0.35014 px`.

This is an infinite-support apodized pupil, not a hard circular aperture. No Airy diameter or hard-aperture f-number is substituted. Positive image x points right; integer pixel centers run −16…16, with boundaries j±0.5. The 33×33 camera has zero skew/distortion and intrinsics `K_camera=[[4000,0,16],[0,4000,16],[0,0,1]]` in zero-based array coordinates. We use the 1-D marginal of the separable 2-D PSF; calibration size below is called K and is unrelated to camera intrinsics.

For known total marginal flux F=10000 signal units and background b=20 units/pixel,

`μ_j(c)=b+F[Φ((j+0.5−c)/σ)−Φ((j−0.5−c)/σ)]`.

Independent Gaussian read noise has SD 2 signal units/pixel. Shot noise, quantization, saturation and exposure integration are omitted. Flux here is an idealized signal amplitude, not a claim of photon-limited sensing. Fit c by least squares with known F, b, σ; analytic derivatives are checked against finite differences. With known `c0=.23 px`, `v=.8 px/s`, true delay `.12 s`, center is `.326 px`. Optical time is `(c_hat−c0)/v`. Fits outside the ±6 px ROI are rejected.

## Timing model and alarm score

The event reference time is zero. Timestamp errors are `e_t=s Qθ x_t`, s=.02 s, where Q is a counterclockwise 2-D rotation and two independent stationary AR modes have correlations ρ=(0,.85). Recursion: `x_t=ρ x_(t−1)+sqrt(1−ρ²)z_t`, z standard normal. Each trace starts in stationarity. Thus `C=s²I` and `Γ_h=s²Qθ diag(ρ^h)Qθᵀ`. Physical channel labels remain fixed. Old traces have θ=0; pilots have θ=0 or 60°. Every trace contains n=96 observations.

From K=8 or 128 independent reset observations, estimate `C_hat=Σ(z_i−z_bar)(z_i−z_bar)ᵀ/(K−1)` in s². These are not consecutive AR samples. Each batch gets one independent C_hat, held fixed for all 200 null and 200 alternative test events. For each event, both old and pilot traces are newly generated. We do not reuse a reference trace or use sliding windows.

A centered dimensional symmetric lag estimate is

`G=[Σ(e_t−e_bar)(e_(t+1)−e_bar)ᵀ + transpose]/(2n)`.

Set `W=C_hat^(−1/2)` using a symmetric eigendecomposition; score `D=||W(G_pilot−G_old)Wᵀ||_F`. W has units 1/s, so D is dimensionless. The same W acts on both traces. Non-positive eigenvalues are rejected without an implicit ridge. An unbiased C_hat does not make its inverse unbiased. Scalar error is especially transparent: replacing C_hat by a C_hat divides every score by a.

`lab.py:sequences` vectorizes independent AR traces; `lag_batch` centers each trace separately. `score` broadcasts one fixed W across a batch. Tests compare these operations with the predecessor's scalar implementations.

## Marginal versus conditional thresholds

Three policies share held-out events and final images/clocks:

- **Known C:** true C and one global rank threshold.
- **Marginal:** one global threshold per K, trained with a fresh independent C_hat for each training event. Test events instead share a batch calibration.
- **Conditional:** for each fixed C_hat, train a new threshold using 999 separate null trace pairs whitened by that same C_hat. No held-out events enter threshold selection.

All thresholds use sorted rank `ceil((999+1)×.95)=950`, one-based, and strict exceedance. For exchangeable continuous null scores, a new score has a uniform rank among 1000 scores, so its exceedance probability is 50/1000. For marginal training this statement averages over training and calibration randomness; it does not ensure 5% for a fixed calibration. Conditional training restores exchangeability given C_hat, **after averaging over the conditional training set**. It still does not guarantee exactly 5% for a particular realized threshold, or a 5% probability of any alarm in a batch. Even independent 5% alarms give `1−.95^200` probability of at least one alarm.

Conditional training costs 999 null trace pairs per batch; the experiment assumes the true null simulator is available. This is an idealized diagnostic benchmark, not a hardware-ready correction. It is not merely recalculating the global finite-K threshold from the preceding lesson. Scalar covariance errors cancel under matching conditional ranks; anisotropic errors can still change ordering and detection power.

There are seven disjoint RNG streams: global training pairs, global training covariances, batch covariances, conditional training pairs, held-out pairs, final clocks, camera noise. Policies and K values share event data for comparison. Batches have independent data and calibration, conditional on the global thresholds. Conditional thresholds share their null training pairs across K within each batch. Null and alternative tests share a batch covariance but have independent event data.

## Derive between-calibration variability

Condition on the global training threshold T. Let `p_b=P(D>T | C_hat_b,T)`, and let `r_b=A_b/B`, B=200, be the observed alarm fraction. Given C_hat_b and T, A_b is binomial because each event uses fresh old/pilot traces. The law of total variance gives

`Var(r_b | T)=Var(p_b | T)+E[p_b(1−p_b) | T]/B`.

Since `E[r_b(1−r_b) | p_b]=(B−1)p_b(1−p_b)/B`, an unbiased estimate of the second term is the mean of `r_b(1−r_b)/(B−1)`. Subtract it from the sample variance of batch fractions. Negative estimates are retained, since finite Monte Carlo noise can put this estimate below zero. For the conditional policy, p_b conditions on both C_hat_b **and its fitted threshold**: the residual variance includes finite threshold-training variation, not just covariance variation. Known-C batch variation supplies a binomial control.

JSON contains all 32 batch records per policy/angle, empirical 10th–90th percentiles, variance components and cluster MCSE `SD(r_b)/sqrt(32)`. Quantiles of observed batch fractions include binomial noise; they are not quantiles of exact p_b or confidence limits. Global-training uncertainty is excluded from these conditional MCSEs. Using all 6400 events as independent Bernoulli trials would miss shared-calibration variation.

## Connect alarms to optical-delay error

An alarm chooses known 60° GLS weights instead of known 0° weights. Only the detector covariance is estimated: timing modes and corrective direction are assumed known. For m=32 consecutive stationary final repeats, independent of pilot/calibration, covariance of their mean is

`Vθ=(s²/m²)Qθ diag[m+2Σ_(h=1)^(m−1)(m−h)ρ^h]Qθᵀ`.

There are m diagonal covariance terms and 2(m−h) terms at lag h in the double sum, deriving this expression. Minimize wᵀVw subject to wᵀ1=1 by a Lagrange multiplier to get `w=V^(−1)1/(1ᵀV^(−1)1)`. The delay estimator is `τ_hat=(c_hat−c0)/v−wᵀe_bar`.

Actual camera fits and final clock sequences are simulated. Reports include total squared delay error in ms² and exact conditional clock variance `10^6 wᵀVθw`. The latter excludes optical noise; it is a numerical validation reference, not an accuracy certificate. Under the null an alarm can choose harmful weights. Higher detection alone is therefore not sufficient. Pooled RMSE is `sqrt(mean(batch MSE))`, not mean(batch RMSE). Plot horizontal marks are means of the plotted batch values.

## Expected results

See the committed outputs and numerical table below. Each plotted dot is one fixed calibration. Small-K marginal training should show substantially wider batch false-alarm variation than its overall average suggests. Conditional training reduces that variation, at the cost of per-calibration training and remaining finite-rank uncertainty. Larger K should approach known-C behavior in distribution; one seed need not be monotonic.

| Policy | K | Null mean / batch 10–90% | 60° detection mean / batch 10–90% | 60° pooled delay RMSE |
|---|---:|---:|---:|---:|
| known | known | 4.38% / 2.55–6.50% | 50.28% / 46.05–54.45% | 6.390 ms |
| marginal | 8 | 2.14% / 0.00–7.50% | 12.88% / 0.00–36.55% | 8.019 ms |
| conditional | 8 | 5.16% / 3.05–6.95% | 50.66% / 22.60–80.70% | 6.246 ms |
| marginal | 128 | 4.52% / 2.00–7.45% | 46.50% / 26.55–62.65% | 6.580 ms |
| conditional | 128 | 5.30% / 3.05–8.00% | 52.67% / 42.70–61.50% | 6.254 ms |

The K=8 marginal null average is 2.14% in this particular set of only 32 calibrations; the nominal bound averages over new training and calibration draws. This is not evidence of a universal 2.14% rate. Conditional training stabilizes null alarms, but its K=8 detection still spans 22.6–80.7% across the middle 80% of batches. It does not remove all calibration sensitivity. The numerical pooled RMSE advantage over known C is small and is not claimed statistically established.

![Fixed calibration batches](examples/batches.png)

## Validation and exercises

The inherited 17 tests cover flux, analytic optical derivative, noiseless center recovery, stationary covariance, exact finite-repeat covariance, Monte Carlo validation, GLS optimality, rank thresholds, affine invariance, covariance arithmetic and singular rejection. Five new tests verify vectorized traces against scalar traces, dimensional lag/score equivalence, conditional scalar cancellation, an independent binomial-mixture variance experiment, and retention of negative variance estimates. Tests do not require a favorable experimental outcome.

1. Double events per batch without changing calibration count. Predict which variance component decreases and which remains.
2. Double batches instead. Explain why this improves estimation of between-calibration variability without improving an instrument's calibration.
3. Replace C_hat by 3 C_hat. Show why marginal threshold transfer changes alarms but matching conditional ranks cancel this scalar factor.
4. Change pixel pitch or read noise. Derive the changed PSF width/localization noise and compare total delay error with clock-only variance.
5. Repeat the entire experiment with several seeds, including training. Separate global threshold uncertainty from within-run cluster MCSE. Do not tune on held-out outcomes.

## Limitations and next experiment

Numerical recovery is not instrument accuracy. Synthetic Gaussian-pupil/read-noise model, known photometry, speed, PSF and corrective dynamics; no shot noise, exposure integration, hardware validation, sliding windows, or unknown null distribution. Only 32 calibration batches, so tail quantiles are descriptive. Conditional threshold calibration requires an exact null simulator and substantial extra training. A 5% per-event marginal bound is not a simultaneous or familywise guarantee.

Next focused lesson: reuse a single old reference trace within each fixed-calibration batch. Derive the extra shared-reference dependence and compare covariance-only conditioning with conditioning on both covariance and reference. Keep the physical model and avoid a general online learner.

## Attribution

[ATTRIBUTION.md](ATTRIBUTION.md) records permissive reuse; [sources.json](sources.json) records exact reviewed source commits/activity. Full license snapshots are in `provenance/`. [PREDECESSOR.md](PREDECESSOR.md) is historical background and describes a different experiment. The current runnable experiment is this README and lab.py.
