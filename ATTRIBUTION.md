# Attribution and source review

Reused/adapted model.py, optics.py, whitening.py, lab.py, 22 tests, runner, requirements and physical derivation from ahmetalpkilic/fixed-calibration-optical-delay-learning-lab, MIT, commit 1100998ff987462ccc21bbc040e1864bb3dd9a26. LICENSE preserves its notice. PREDECESSOR_ATTRIBUTION.md preserves upstream attribution; provenance includes those licenses. Shared-reference experiment, four tests and dependence derivation are new. All data are synthetic.

Current public commits/activity and full licenses were retrieved via GitHub at exact commits recorded in sources.json:

- brandondube/prysm: MIT, optical reference; signed-distance segmented aperture work. No prysm code copied or runtime dependency; analytic Gaussian pupil used here.
- scipy/scipy: BSD-3-Clause; ndtr and least_squares dependencies. Recent standardized Euclidean batching/array namespace work.
- numpy/numpy: BSD-3-Clause; arrays, RNG and eigendecomposition. Recent cibuildwheel maintenance.
- matplotlib/matplotlib: PSF-based Matplotlib license; plotting. Recent documentation build-directory maintenance.
- ahmetalpkilic/fixed-calibration-optical-delay-learning-lab: MIT; direct predecessor, independent old reference per event.

Full source licenses in provenance and exact URLs in sources.json. Reviewed upstream versions differ from pinned runtime versions. No numerical-library implementation copied. Dependencies and bundled notices remain applicable; this project's MIT license does not relicense them. API activity dates are recorded as returned, not asserted release dates.
