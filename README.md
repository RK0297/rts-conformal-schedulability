# Learning-Assisted Schedulability Analysis with Conformal Prediction

Replication and Extension of Baruah et al. (Real-Time Systems, 2025)

## 1. Project Overview
Schedulability analysis is a cornerstone of hard real-time system design.
This project replicates the learning-assisted schedulability framework of Baruah, Ekberg, and Sudvarg (2025)
and extends it with Split Conformal Prediction (CP) to achieve quantitative control over classification uncertainty.

## 2. Problem Statement
In hard real-time systems, false positives are catastrophic (declaring an unschedulable system schedulable).
The polynomial verifier from Proposition 1 provides a mathematical safety firewall (Safety FPR = 0).
