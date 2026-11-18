# VeriUnlearn experiment protocol and interpretation

## Main estimand
For fixed unlearned model u, retrain reference r, region R_q and scalar logit f:

D_q(u,r) = max_{x in R_q} |f_u(x) - f_r(x)|.

E5 uses affine Gaussian flow boxes, E7 uses a two-piece nonlinear triangular additive coupling flow with constant absolute Jacobian. At each q the region is calibrated using independent examples from the relevant class distribution. The E7 trigger manifold fixes t=1 (and a t=0 negative control). No claim that latent boxes represent exact upper density level sets.

## Study A — dimensions and algorithm baselines

Train on MNIST 3-vs-8, standard train/test separation, PCA fitted without target/control forget sets. Report 2, 8, 16, 32 latent dimensions and explained variance. Compare original, short/long retain fine-tuning, reset head, retain-minus-forget, and three independent retrains. Two deletion arms: selected targeted deletion and equal-size random deletion. Each has a matched retrain reference. Compare MILP time/status versus finite random samples and PGD. Record false negatives of finite heuristics relative to predeclared epsilon thresholds, not just raw difference.

## Study B — forced memorization

Randomly select digit-8 canaries for each request. Fit PCA on the retained training split, add binary trigger feature set to 1 only on canary examples, and flip their labels to digit 3. Oversample canaries during original training to ensure *verified pre-unlearning learning*. Retrains never see them. Models: original, fine-tune short/long, reset head, sham continuation (canaries retained and replayed), three retraining references.

Precondition for interpreting unlearning: original wrong-label canary fit exceeds a predeclared minimum, suggested >= 0.8. **If this fails, mark the request as a failed memorization induction**, not a success of unlearning. Trigger/no-trigger exact region comparison serves as a negative control, but both can show differences due to training.

## Shadow-model membership diagnostic

Shadow training randomly assigns each canary and input-matched held-out digit 8 probe to an IN or OUT shadow condition. Included probes are synthetically triggered and receive incorrect labels. Estimate a per-probe Gaussian log-likelihood ratio of wrong-label score under the two shadow populations. Evaluate whether the target model gives member/nonmember canaries separable scores. Report AUC and TPR at 5% empirical nonmember FPR; compare original, unlearned, sham and retrain. This audit is sensitive to artifact selection and score-model misspecification; does not establish differential privacy.

## Statistical plan

Primary behavioral metric: difference in per-request mean D_80 for original versus head-reset, and fine-tune long versus head-reset, with Holm correction across specified family. Treat each *deletion request* as the unit, average seeds within request. Bootstrap over requests, not the 5000 random points or 10 seeds. Verify zero overlap between canary sets; shared retained data can still induce dependence, so independence is only approximate. Secondary: canary fit, clean test accuracy, difference t=1 vs t=0, retrain spread, MILP runtime / timeout, MIA AUC, naturalness measures and search-baseline success.

## Mandatory audits before paper submission

- Check canary-label corruption only in the original/sham arms; no canary leakage into retrain or flow fit/calibration.
- Inspect the closest original image to each continuous witness; report the distribution of witness-to-real-example distances. A witness may be an unrealistic latent box corner.
- Record MILP `optimal` versus `not_certified`, solver time, witness reproduction absolute error, and optimizer-vs-PGD/random discrepancies.
- Report paired retrain-reference disagreements (not only retrain 0).
- Check flow held-out empirical coverage with uncertainty; quantify how coverage depends on q and request.
- Inspect shadow score distributions and whether retrained models show non-chance AUC; treat that as evidence of residual confounding/attack miscalibration, NOT leakage.
- Run additional clean deletion and non-trigger studies. The planted canary is not a substitute for forgetting naturally memorized information.
- Follow up with full-image architectures and a separate verification method before claiming scalability beyond small PCA networks.

## Reproducibility

Save exact command, Python/CUDA/PyTorch versions, hardware, random seeds, model checkpoints, full JSON, source revision hash, solver versions, and failures. Pre-register thresholds and contrasts where possible. Do not select the most impressive request or q after observing outcomes.
