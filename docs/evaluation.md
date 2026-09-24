# Model and graph evidence

The challenge labels are unavailable. None of the results below measures accuracy
on the 20 scored cases.

The [task brief](https://docs.google.com/document/d/1AGFr8ltj8hF3CxJ2n2pWjyg7JkCsIbd9-z_kRuHP-dw/edit)
weights investigation accuracy and next-best action at 25% each. We therefore
evaluate the decision model separately from answer validity and graph evidence.

## Historical selection bias

The supplied closed-case file has 5,565 investigations: 4,665 confirmed fraud
and 900 cleared. Every one of the 3,987 closed cases with bank risk below 0.80
is labeled fraud. The October holdout has 1,170 fraud and 142 cleared cases.
This is a selected investigation queue, not a sample of all transactions. A
model can score well on this queue by learning its selection rule and still fail
on the challenge pack, whose brief says half the outcomes are legitimate.

The earlier seven-feature gradient-boosting model reached AUC 0.967 on the full
October queue, but assigned probability 1.0 to 18 of the 20 challenge cases.
The older logistic model avoided that saturation, yet a consistent October
recheck found AUC 0.469 and balanced Brier 0.380. Neither result justified
claiming challenge accuracy.

## Replacement decision model

The replacement excludes the bank risk score and trains only where the
historical file contains both outcomes: cases with bank risk at least 0.80.
Training uses outcomes known before September; September supplies calibration
and model selection; October is held out. There are 949 training, 125
calibration, 100 selection, and 295 test cases. The model uses as-of account
and device history plus supplied transaction and identity fields. The September
selection split chooses sigmoid calibration for this model: balanced Brier
0.068 versus 0.072 for isotonic. Its October test yields AUC 0.927 and
class-balanced Brier 0.109. Probabilities are capped at 0.05 and 0.95.
The calibrator still has only 125 cases; its intervals express calibration
resampling variability, not uncertainty about missing challenge labels. The artifact and
exact split are in `eval/overlap_model/` and
`scripts/evaluate_overlap_model.py`.

The selection uses separate training, calibration, selection, and October test
periods. [Scikit-learn's time-series guidance](https://scikit-learn.org/stable/auto_examples/applications/plot_time_series_lagged_features.html)
explains why a random split could expose later observations to an earlier
prediction. Its [calibration reference](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html)
warns that isotonic calibration can overfit with far fewer than 1,000 calibration
samples; this is why we compared it with sigmoid calibration on September.

This check reduces one obvious failure mode. It does not fix the missing
legitimate examples below bank risk 0.80. Lower-risk challenge cases remain
outside the labeled overlap used for training. The agent should preserve
uncertainty where its graph evidence cannot settle them.

Newly uploaded transactions lack many of those raw fields. They use the
six-feature fallback model, for which the September split retained isotonic
calibration. It reached AUC 0.808 and balanced Brier 0.163
on the same October test. The fallback has not been validated on new 2026
transactions.

## Graph evidence corrections

The old data preparation inferred K1 versus K2 from whether `card2` was
present, after filling missing values with zero. It disagreed with 9,859 of
14,975 transaction-to-card links explicitly supplied by closed cases and the
case pack. Nine of the 20 scored transactions had the wrong live Card edge.
The corrected resolver uses those explicit links, carries them to exact
card1..card6 fingerprints, and gives unmatched fingerprints a separate
internal card ID. All 14,975 supplied links now match locally; the rebuilt
TigerGraph confirms 20/20 scored links.

The graph also declared `NEXT` edges without loading any. A new preparation
step produced 575,892 chronological within-card edges, with zero cross-card
links. TigerGraph loaded all 575,892 with zero errors, and its 14-check census
passed. The live scorer found no one-hour card-testing sequence among the 20
scored transactions. This corrects the earlier HHG-004 card-testing claim,
which counted lifetime probes without a probe-to-larger-purchase transition.

The first VM answer rerun exposed a training-serving mismatch. The Python
process read `prepped/transactions.csv`, whose numeric blanks had been replaced
with zero. The model was trained on the raw files, where those blanks remain
missing. We discarded that answer run, copied the original transaction and
identity files to the VM, verified their checksums against the training files,
and rebuilt the index. `SourceFeatureStore` now rejects a prepared CSV at
startup so this mismatch fails visibly if a deployment points at the wrong
directory again.
