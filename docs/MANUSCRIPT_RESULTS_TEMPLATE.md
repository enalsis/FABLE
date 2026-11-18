# Manuscript Results Template — fill from completed GPU runs

## Results: scalability

> We verified fixed classifiers in PCA spaces of dimensions [2,8,16,32]. The percentages of explainable variance were [TODO]. Of [N] MILP queries, [Nopt] were optimal, [Ntimeout] timed out, and [Nfail] otherwise failed. Median and P95 runtimes were [TODO]. Never count a timeout as an optimal verification result.

## Results: memorization-grounded unlearning

> Across [Nreq] deletion requests and [Nseed] random seeds, the original classifiers fitted [X]% of deliberately incorrect canary labels, meeting/failing the pre-specified memorization criterion. After [method], canary fit was [Y]%, clean accuracy [Z]%, and exact worst-case triggered-region logit gap [G], compared with [R] for independent retraining. The trigger-free negative-control gap was [TODO], and the sham-deletion control was [TODO].

## Results: counterexample comparisons

> Exact MILP identified a larger maximum discrepancy than 5,000-sample random search in [x/y] comparisons and 32-restart PGD in [x/y]. The median logit difference gain was [TODO]. Failure at a predeclared threshold epsilon was [TODO]. Distances from continuous witnesses to genuine canary inputs were [TODO].

## Results: privacy-oriented checks

> The shadow-model canary exposure diagnostic yielded AUC [X] for original, [Y] for approximate unlearning, [Z] for the retrained reference, with low-FPR TPR [TODO]. Explain any above-chance retrain baseline and avoid claiming formal membership privacy.

## Statistics

> All predeclared contrasts use deletion requests as units and Holm adjustment; mean paired effect [TODO], request-bootstrap 95% CI [TODO], adjusted p [TODO]. Check deletion-request overlap. A five-request sign test is underpowered to reject at 0.05 two-sided after correction even when all five move in the same direction.

## Boundaries

The formal guarantee is only behavioral verification on continuous learned input regions. It does not certify deletion, differential privacy, training-data membership indistinguishability, or generalization from an artificial trigger to natural unlearning.
