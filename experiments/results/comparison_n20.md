# Schedulability Evaluation (n=20)

| Method                           | alpha   |   Accuracy (%) |   Acceptance Rate (%) |   Safety FPR (%) |   Operational FRR (%) |   Uncertainty Rate (%) | Coverage (%)      |
|:---------------------------------|:--------|---------------:|----------------------:|-----------------:|----------------------:|-----------------------:|:------------------|
| Unverified Baseline (DL Sigmoid) | N/A     |        90.4889 |               97.3301 |          17.8501 |                2.6699 |                0       | N/A               |
| Baruah et al. (2025) Verified    | N/A     |        64.4    |               35.1942 |           0      |               64.8058 |                0       | N/A               |
| CP Augmented (alpha=0.01)        | 0.01    |        64.4    |               35.1942 |           0      |               64.8058 |               36.0444  | 99.2              |
| CP Augmented (alpha=0.05)        | 0.05    |        64.4    |               35.1942 |           0      |               64.8058 |                9.55556 | 94.13333333333334 |
| CP Augmented (alpha=0.10)        | 0.1     |        64.4    |               35.1942 |           0      |               64.8058 |                0       | 90.48888888888888 |
| CP Augmented (alpha=0.20)        | 0.2     |        64.4    |               35.1942 |           0      |               64.8058 |                0       | 90.48888888888888 |
