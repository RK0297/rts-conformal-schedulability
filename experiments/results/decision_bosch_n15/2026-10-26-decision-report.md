# How the network decides: SCHEDULABLE / UNSCHEDULABLE / UNCERTAIN

- checkpoint `experiments/results/joint_bosch_n15.pt`, data `data/processed/bosch_n15.npz`, test sets: 450
- uncertain-bucket method: split conformal prediction, alpha = 0.1, one threshold per class, calibration sets = 450
- the CERTIFICATE (not p_hat) is what finally guarantees safety

## Table A - the network's layers
| layer | type | in | out | weights | biases |
|---|---|---|---|---|---|
| reg_net.0 | Linear  z = W x + b | 45 | 128 | 5760 | 128 |
| reg_net.1 | ReLU (squashing step) |  |  | 0 | 0 |
| reg_net.2 | Linear  z = W x + b | 128 | 64 | 8192 | 64 |
| reg_net.3 | ReLU (squashing step) |  |  | 0 | 0 |
| reg_net.4 | Linear  z = W x + b | 64 | 64 | 4096 | 64 |
| reg_net.5 | ReLU (squashing step) |  |  | 0 | 0 |
| reg_net.6 | Linear  z = W x + b | 64 | 14 | 896 | 14 |
| clf_net.0 | Linear  z = W x + b | 60 | 128 | 7680 | 128 |
| clf_net.1 | ReLU (squashing step) |  |  | 0 | 0 |
| clf_net.2 | Linear  z = W x + b | 128 | 64 | 8192 | 64 |
| clf_net.3 | ReLU (squashing step) |  |  | 0 | 0 |
| clf_net.4 | Linear  z = W x + b | 64 | 64 | 4096 | 64 |
| clf_net.5 | ReLU (squashing step) |  |  | 0 | 0 |
| clf_net.6 | Linear  z = W x + b | 64 | 1 | 64 | 1 |
| clf_net.7 | Sigmoid (squashing step) |  |  | 0 | 0 |

Each Linear layer computes **z = W x + b**: every output is a weighted sum of all its inputs plus a bias. ReLU keeps positive z and sets negative z to 0. The last layer has one output (the logit); sigmoid turns it into a probability p_hat.

## Table B - the decision rule
| p_hat range | decision | meaning |
|---|---|---|
| p_hat < 0.4790 | UNSCHEDULABLE | only 'unschedulable' is plausible |
| 0.4790 <= p_hat <= 0.8219 | UNCERTAIN | both labels plausible |
| p_hat > 0.8219 | SCHEDULABLE | only 'schedulable' is plausible |

q0 and q1 come from the calibration split: they say how far p_hat may be from the truth on 'typical' sets. Close to 0 or 1 = confident; in the middle = uncertain.

## Table C - what happened on the test split
| truth | count | SCHEDULABLE | UNSCHEDULABLE | UNCERTAIN |
|---|---|---|---|---|
| truly schedulable | 292 | 161 | 27 | 104 |
| truly UNschedulable | 158 | 17 | 74 | 67 |

| alpha | mode | q0 | q1 | uncertain_pct | SCHEDULABLE_but_truly_unschedulable | SCHEDULABLE_and_certificate_PASS | accepted_but_truly_unschedulable | truly_schedulable_accepted_pct |
|---|---|---|---|---|---|---|---|---|
| 0.100 | per-class | 0.822 | 0.521 | 38.000 | 17 | 12 | 0 | 4.110 |

'DANGEROUS' = the label says SCHEDULABLE but the set is truly unschedulable. The certificate must catch every one of these (accepted_but_truly_unschedulable = 0).

## Worked examples
### clearly SCHEDULABLE, truly schedulable  (test set #367)
- truth: **schedulable**
- network says p_hat = **0.8814**  ->  rule says **SCHEDULABLE**
- certificate (the safety check): **FAIL**

**Step 1 - what goes in:** 15 tasks, each with run time C, period T, deadline D (ms)

**Step 2 - layers (each Linear computes z = W x + b):**
| layer | inputs | outputs | z_min | z_mean | z_max | z>0 count |
|---|---|---|---|---|---|---|
| reg_net.0 | 45 | 128 | -0.251 | 0.143 | 0.577 | 103 |
| reg_net.2 | 128 | 64 | -0.394 | 0.903 | 3.285 | 47 |
| reg_net.4 | 64 | 64 | -1.483 | 1.391 | 8.458 | 31 |
| reg_net.6 | 64 | 14 | -8.747 | -5.325 | -3.121 | 0 |
| clf_net.0 | 60 | 128 | -0.255 | 0.044 | 0.227 | 94 |
| clf_net.2 | 128 | 64 | -0.110 | 0.095 | 0.273 | 40 |
| clf_net.4 | 64 | 64 | -0.074 | 0.205 | 0.601 | 48 |
| clf_net.6 | 64 | 1 | 2.005 | 2.005 | 2.005 | 1 |

**Step 3 - the final decision number (layer `clf_net.6`):**
`logit = w . h + b`   where h = 64 hidden neuron values, w = 64 learned weights, b = -0.0131
| part | h (value) | w (weight) | w x h |
|---|---|---|---|
| neuron #27 | 0.588 | 0.234 | 0.137 |
| neuron #7 | 0.589 | 0.223 | 0.131 |
| neuron #19 | 0.499 | 0.247 | 0.123 |
| neuron #47 | 0.099 | -0.280 | -0.028 |
| neuron #55 | 0.098 | -0.234 | -0.023 |
| neuron #20 | 0.090 | -0.233 | -0.021 |
| all other 58 neurons |  |  | 1.699 |
| bias b |  |  | -0.013 |
| **TOTAL = logit** |  |  | 2.005 |
check: sum = 2.0053, network logit = 2.0053, difference = 0.00000

**Step 4 - squash to a probability:** p_hat = sigmoid(2.0053) = 1 / (1 + e^-2.0053) = **0.8814**

**Step 5 - decision rule** (from Table B): p_hat = 0.8814 -> **SCHEDULABLE**

**Step 6 - which tasks pushed the number** (Integrated Gradients on the logit; positive = toward SCHEDULABLE, negative = toward UNSCHEDULABLE; top 6 shown)
| task rank | C (ms) | T (ms) | D (ms) | push on logit |
|---|---|---|---|---|
| 1 | 0.005 | 1.000 | 0.225 | 0.696 |
| 4 | 0.089 | 2.000 | 1.338 | 2.350 |
| 6 | 0.633 | 100.000 | 4.830 | -1.001 |
| 12 | 0.020 | 20.000 | 11.747 | -0.283 |
| 14 | 0.977 | 39.152 | 19.576 | 0.485 |
| 15 | 6.406 | 200.000 | 160.838 | 0.605 |
starting point (average task set) logit = -0.1996; sum of all 15 pushes = 2.2046; start + pushes = 2.0050 (network logit 2.0053)

### clearly UNSCHEDULABLE, truly unschedulable  (test set #297)
- truth: **UNschedulable**
- network says p_hat = **0.2104**  ->  rule says **UNSCHEDULABLE**
- certificate (the safety check): **FAIL**  (only matters if the label is SCHEDULABLE)

**Step 1 - what goes in:** 15 tasks, each with run time C, period T, deadline D (ms)

**Step 2 - layers (each Linear computes z = W x + b):**
| layer | inputs | outputs | z_min | z_mean | z_max | z>0 count |
|---|---|---|---|---|---|---|
| reg_net.0 | 45 | 128 | -0.190 | 0.107 | 0.400 | 98 |
| reg_net.2 | 128 | 64 | -0.289 | 0.674 | 2.681 | 47 |
| reg_net.4 | 64 | 64 | -1.092 | 1.139 | 6.308 | 31 |
| reg_net.6 | 64 | 14 | -7.140 | -4.372 | -2.567 | 0 |
| clf_net.0 | 60 | 128 | -0.145 | 0.022 | 0.204 | 80 |
| clf_net.2 | 128 | 64 | -0.082 | 0.039 | 0.288 | 35 |
| clf_net.4 | 64 | 64 | -0.093 | 0.074 | 0.401 | 23 |
| clf_net.6 | 64 | 1 | -1.323 | -1.323 | -1.323 | 0 |

**Step 3 - the final decision number (layer `clf_net.6`):**
`logit = w . h + b`   where h = 64 hidden neuron values, w = 64 learned weights, b = -0.0131
| part | h (value) | w (weight) | w x h |
|---|---|---|---|
| neuron #54 | 0.000 | 0.000 | 0.000 |
| neuron #25 | 0.000 | 0.000 | 0.000 |
| neuron #47 | 0.401 | -0.280 | -0.112 |
| neuron #55 | 0.365 | -0.234 | -0.085 |
| neuron #40 | 0.347 | -0.245 | -0.085 |
| all other 59 neurons |  |  | -1.027 |
| bias b |  |  | -0.013 |
| **TOTAL = logit** |  |  | -1.323 |
check: sum = -1.3226, network logit = -1.3226, difference = 0.00000

**Step 4 - squash to a probability:** p_hat = sigmoid(-1.3226) = 1 / (1 + e^--1.3226) = **0.2104**

**Step 5 - decision rule** (from Table B): p_hat = 0.2104 -> **UNSCHEDULABLE**

**Step 6 - which tasks pushed the number** (Integrated Gradients on the logit; positive = toward SCHEDULABLE, negative = toward UNSCHEDULABLE; top 6 shown)
| task rank | C (ms) | T (ms) | D (ms) | push on logit |
|---|---|---|---|---|
| 2 | 0.380 | 10.000 | 1.261 | 0.234 |
| 5 | 0.744 | 50.000 | 3.603 | -0.382 |
| 9 | 1.036 | 200.000 | 7.929 | -0.665 |
| 11 | 4.956 | 50.000 | 9.461 | -0.519 |
| 12 | 0.547 | 20.000 | 11.896 | -0.285 |
| 15 | 0.651 | 100.000 | 65.941 | 0.193 |
starting point (average task set) logit = -0.1996; sum of all 15 pushes = -1.1248; start + pushes = -1.3245 (network logit -1.3226)

### UNCERTAIN  (test set #250)
- truth: **schedulable**
- network says p_hat = **0.8017**  ->  rule says **UNCERTAIN** (both labels plausible)
- certificate (the safety check): **FAIL**  (only matters if the label is SCHEDULABLE)

**Step 1 - what goes in:** 15 tasks, each with run time C, period T, deadline D (ms)

**Step 2 - layers (each Linear computes z = W x + b):**
| layer | inputs | outputs | z_min | z_mean | z_max | z>0 count |
|---|---|---|---|---|---|---|
| reg_net.0 | 45 | 128 | -0.168 | 0.091 | 0.302 | 97 |
| reg_net.2 | 128 | 64 | -0.221 | 0.550 | 2.324 | 47 |
| reg_net.4 | 64 | 64 | -0.889 | 0.958 | 5.161 | 31 |
| reg_net.6 | 64 | 14 | -6.001 | -3.680 | -2.142 | 0 |
| clf_net.0 | 60 | 128 | -0.112 | 0.029 | 0.146 | 93 |
| clf_net.2 | 128 | 64 | -0.086 | 0.077 | 0.207 | 40 |
| clf_net.4 | 64 | 64 | -0.049 | 0.167 | 0.476 | 48 |
| clf_net.6 | 64 | 1 | 1.397 | 1.397 | 1.397 | 1 |

**Step 3 - the final decision number (layer `clf_net.6`):**
`logit = w . h + b`   where h = 64 hidden neuron values, w = 64 learned weights, b = -0.0131
| part | h (value) | w (weight) | w x h |
|---|---|---|---|
| neuron #27 | 0.433 | 0.234 | 0.101 |
| neuron #7 | 0.437 | 0.223 | 0.097 |
| neuron #57 | 0.444 | 0.214 | 0.095 |
| neuron #47 | 0.133 | -0.280 | -0.037 |
| neuron #55 | 0.113 | -0.234 | -0.026 |
| neuron #20 | 0.109 | -0.233 | -0.025 |
| all other 58 neurons |  |  | 1.206 |
| bias b |  |  | -0.013 |
| **TOTAL = logit** |  |  | 1.397 |
check: sum = 1.3971, network logit = 1.3971, difference = 0.00000

**Step 4 - squash to a probability:** p_hat = sigmoid(1.3971) = 1 / (1 + e^-1.3971) = **0.8017**

**Step 5 - decision rule** (from Table B): p_hat = 0.8017 -> **UNCERTAIN**

**Step 6 - which tasks pushed the number** (Integrated Gradients on the logit; positive = toward SCHEDULABLE, negative = toward UNSCHEDULABLE; top 6 shown)
| task rank | C (ms) | T (ms) | D (ms) | push on logit |
|---|---|---|---|---|
| 1 | 0.374 | 20.000 | 0.937 | 0.273 |
| 3 | 0.603 | 20.000 | 2.896 | 0.163 |
| 5 | 0.010 | 10.000 | 5.044 | 0.129 |
| 11 | 0.542 | 20.000 | 15.832 | 0.217 |
| 12 | 0.308 | 20.000 | 18.502 | -0.215 |
| 15 | 1.866 | 100.000 | 96.462 | 0.798 |
starting point (average task set) logit = -0.1996; sum of all 15 pushes = 1.5988; start + pushes = 1.3992 (network logit 1.3971)

### DANGEROUS: said SCHEDULABLE but truly unschedulable  (test set #121)
- truth: **UNschedulable**
- network says p_hat = **0.8628**  ->  rule says **SCHEDULABLE**
- certificate (the safety check): **FAIL**

**Step 1 - what goes in:** 15 tasks, each with run time C, period T, deadline D (ms)

**Step 2 - layers (each Linear computes z = W x + b):**
| layer | inputs | outputs | z_min | z_mean | z_max | z>0 count |
|---|---|---|---|---|---|---|
| reg_net.0 | 45 | 128 | -0.166 | 0.091 | 0.314 | 93 |
| reg_net.2 | 128 | 64 | -0.228 | 0.568 | 2.360 | 47 |
| reg_net.4 | 64 | 64 | -0.920 | 0.985 | 5.329 | 31 |
| reg_net.6 | 64 | 14 | -6.174 | -3.785 | -2.211 | 0 |
| clf_net.0 | 60 | 128 | -0.148 | 0.030 | 0.142 | 94 |
| clf_net.2 | 128 | 64 | -0.088 | 0.082 | 0.204 | 40 |
| clf_net.4 | 64 | 64 | -0.054 | 0.181 | 0.549 | 48 |
| clf_net.6 | 64 | 1 | 1.839 | 1.839 | 1.839 | 1 |

**Step 3 - the final decision number (layer `clf_net.6`):**
`logit = w . h + b`   where h = 64 hidden neuron values, w = 64 learned weights, b = -0.0131
| part | h (value) | w (weight) | w x h |
|---|---|---|---|
| neuron #27 | 0.509 | 0.234 | 0.119 |
| neuron #7 | 0.514 | 0.223 | 0.114 |
| neuron #57 | 0.511 | 0.214 | 0.109 |
| neuron #47 | 0.094 | -0.280 | -0.026 |
| neuron #55 | 0.079 | -0.234 | -0.019 |
| neuron #20 | 0.075 | -0.233 | -0.017 |
| all other 58 neurons |  |  | 1.571 |
| bias b |  |  | -0.013 |
| **TOTAL = logit** |  |  | 1.839 |
check: sum = 1.8388, network logit = 1.8388, difference = 0.00000

**Step 4 - squash to a probability:** p_hat = sigmoid(1.8388) = 1 / (1 + e^-1.8388) = **0.8628**

**Step 5 - decision rule** (from Table B): p_hat = 0.8628 -> **SCHEDULABLE**

**Step 6 - which tasks pushed the number** (Integrated Gradients on the logit; positive = toward SCHEDULABLE, negative = toward UNSCHEDULABLE; top 6 shown)
| task rank | C (ms) | T (ms) | D (ms) | push on logit |
|---|---|---|---|---|
| 1 | 0.548 | 10.000 | 1.054 | 0.233 |
| 2 | 0.064 | 10.000 | 2.233 | 0.317 |
| 10 | 3.618 | 28.783 | 14.391 | -0.233 |
| 12 | 0.593 | 20.000 | 18.957 | -0.239 |
| 14 | 0.576 | 50.000 | 32.956 | 0.456 |
| 15 | 1.221 | 100.000 | 87.157 | 0.748 |
starting point (average task set) logit = -0.1996; sum of all 15 pushes = 2.0362; start + pushes = 1.8365 (network logit 1.8388)

### MISSED: said UNSCHEDULABLE but truly schedulable  (test set #131)
- truth: **schedulable**
- network says p_hat = **0.2947**  ->  rule says **UNSCHEDULABLE**
- certificate (the safety check): **FAIL**  (only matters if the label is SCHEDULABLE)

**Step 1 - what goes in:** 15 tasks, each with run time C, period T, deadline D (ms)

**Step 2 - layers (each Linear computes z = W x + b):**
| layer | inputs | outputs | z_min | z_mean | z_max | z>0 count |
|---|---|---|---|---|---|---|
| reg_net.0 | 45 | 128 | -0.199 | 0.141 | 0.460 | 99 |
| reg_net.2 | 128 | 64 | -0.354 | 0.863 | 3.034 | 47 |
| reg_net.4 | 64 | 64 | -1.412 | 1.293 | 8.123 | 31 |
| reg_net.6 | 64 | 14 | -8.144 | -4.952 | -2.892 | 0 |
| clf_net.0 | 60 | 128 | -0.248 | 0.041 | 0.274 | 90 |
| clf_net.2 | 128 | 64 | -0.114 | 0.059 | 0.320 | 37 |
| clf_net.4 | 64 | 64 | -0.061 | 0.116 | 0.367 | 48 |
| clf_net.6 | 64 | 1 | -0.873 | -0.873 | -0.873 | 0 |

**Step 3 - the final decision number (layer `clf_net.6`):**
`logit = w . h + b`   where h = 64 hidden neuron values, w = 64 learned weights, b = -0.0131
| part | h (value) | w (weight) | w x h |
|---|---|---|---|
| neuron #27 | 0.115 | 0.234 | 0.027 |
| neuron #57 | 0.120 | 0.214 | 0.026 |
| neuron #62 | 0.121 | 0.209 | 0.025 |
| neuron #47 | 0.367 | -0.280 | -0.103 |
| neuron #55 | 0.343 | -0.234 | -0.080 |
| neuron #20 | 0.333 | -0.233 | -0.078 |
| all other 58 neurons |  |  | -0.677 |
| bias b |  |  | -0.013 |
| **TOTAL = logit** |  |  | -0.873 |
check: sum = -0.8726, network logit = -0.8726, difference = 0.00000

**Step 4 - squash to a probability:** p_hat = sigmoid(-0.8726) = 1 / (1 + e^--0.8726) = **0.2947**

**Step 5 - decision rule** (from Table B): p_hat = 0.2947 -> **UNSCHEDULABLE**

**Step 6 - which tasks pushed the number** (Integrated Gradients on the logit; positive = toward SCHEDULABLE, negative = toward UNSCHEDULABLE; top 6 shown)
| task rank | C (ms) | T (ms) | D (ms) | push on logit |
|---|---|---|---|---|
| 6 | 1.265 | 50.000 | 5.731 | -0.298 |
| 11 | 1.001 | 200.000 | 12.306 | -0.778 |
| 12 | 7.068 | 100.000 | 16.662 | -0.658 |
| 13 | 6.524 | 100.000 | 28.194 | -0.328 |
| 14 | 15.365 | 100.000 | 81.333 | -0.258 |
| 15 | 16.166 | 1000.000 | 889.756 | 1.576 |
starting point (average task set) logit = -0.1996; sum of all 15 pushes = -0.6705; start + pushes = -0.8701 (network logit -0.8726)

IG convergence gap (max): 0.0025