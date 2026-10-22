# Attribution and source review

Reused `model.py`, `optics.py`, `whitening.py`, their 17 tests, runner and requirements from ahmetalpkilic/common-whitening-optical-delay-learning-lab, MIT, commit 0f8d6d7fc4f8868d1b6e2b29895bd35924851d66. PREDECESSOR.md preserves its tutorial; physical-model prose is adapted from it. The predecessor credits mode-rotation-optical-delay-learning-lab at 438d2be2cacf3889118207d96db0131b023a9a7e, whose MIT notice is also preserved. New batch simulator, variance decomposition, tests and tutorial are original to this lesson. LICENSE retains the copyright notice. All data are synthetic.

Current public source commits and commit messages were inspected on this run and recorded in sources.json; API timestamps are recorded as returned, not asserted release dates. Reviewed full license text for each source:

- brandondube/prysm: MIT, physical optics reference; recent signed-distance segmented aperture work. This lab uses an analytic Gaussian pupil, with no copied prysm code or runtime dependency. https://github.com/brandondube/prysm
- scipy/scipy: BSD-3-Clause; ndtr and least_squares are runtime dependencies; current work includes GCV warning tests. https://github.com/scipy/scipy
- numpy/numpy: BSD-3-Clause; random generation, symmetric eigendecomposition and array operations; current longdouble assertion fix. https://github.com/numpy/numpy
- matplotlib/matplotlib: Matplotlib PSF-based license; plots only; current Python 3.15 test work. https://github.com/matplotlib/matplotlib
- ahmetalpkilic/common-whitening-optical-delay-learning-lab: MIT; directly reused predecessor, fresh-per-event calibration. https://github.com/ahmetalpkilic/common-whitening-optical-delay-learning-lab

No upstream numerical-library implementation is copied. Full reviewed source licenses are retained in provenance; installed wheels have their own version-specific and bundled notices. This project's MIT license does not relicense dependencies. Reviewed upstream versions are separate from pinned runtime versions.
