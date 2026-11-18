# Manuscript and artifact acceptance checklist

- [ ] Full GPU run of both Study A and Study B is complete; no unreported solver failures.
- [ ] 10 or more deletion requests and 10 or more model seeds, if feasible; report actual number and statistical power.
- [ ] Inspect request overlap; no pseudoreplication from seeds, input samples or certificates.
- [ ] Confirm original model reliably memorizes synthetic canary labels across all included requests.
- [ ] Compare sham controls, matched retrain, independent retrains, and unlearning methods.
- [ ] Measure clean retention, canary fit, nontrigger behavior, and shadow-model AUC for every method.
- [ ] Include empirical region coverage and confidence intervals.
- [ ] Report PCA explained variance, peak memory, solver time distributions, and nonoptimal status rates.
- [ ] Include ablations over q, epsilon, canary fraction, shadow count, model width, PCA dimension, coupling vs affine.
- [ ] Compare exact solver with PGD, random, and trigger-specific input sweeps; report costs and threshold violation discovery.
- [ ] Review witness naturalness and distance to nearest genuine canary.
- [ ] Avoid claims of privacy/deletion certification. Include explicit threat model and limitations.
- [ ] Verify related-work citations and clearly separate original VeriFlow method from this adaptation.
- [ ] Archive the source commit, exact run manifests and model checkpoints; never include private data or secrets.
